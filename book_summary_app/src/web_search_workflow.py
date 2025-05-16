from typing import TypedDict, Annotated, Sequence
from langgraph.graph import Graph, StateGraph
from langchain_core.messages import HumanMessage, AIMessage
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from datetime import datetime
import os
from openai import OpenAI
import json
import requests
import re

# .envファイルから環境変数を読み込む（book_summary.pyですでに読み込んでいる場合は影響なし）
try:
    from dotenv import load_dotenv
    from pathlib import Path
    root_dir = Path(__file__).resolve().parent.parent
    env_path = root_dir / ".env"
    if env_path.exists():
        load_dotenv(env_path)
except ImportError:
    pass  # book_summary.pyで警告を表示しているので、ここでは省略

# OpenAI クライアント初期化
api_key = os.environ.get("OPENAI_API_KEY")
if not api_key:
    print("警告: OPENAI_API_KEYが設定されていません。.envファイルまたは環境変数で設定してください。")
client = OpenAI(api_key=api_key)

# 状態の型定義
class WorkflowState(TypedDict):
    messages: Annotated[Sequence[HumanMessage | AIMessage], "会話履歴"]
    book_title: Annotated[str, "検索する書籍タイトル"]
    search_results: Annotated[list, "検索結果"]
    initial_summary: Annotated[str, "初期要約"]
    agent_feedback: Annotated[str, "フィードバック"]
    final_summary: Annotated[str, "最終的な要約"]

# 検索クエリを最適化するエージェント
def create_query_optimizer_agent():
    def query_optimizer_agent(state: WorkflowState) -> WorkflowState:
        book_title = state["book_title"]
        
        # 検索クエリを最適化
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "あなたは検索クエリ最適化の専門家です。与えられた書籍タイトルに基づいて、書評や要約情報を効率的に検索するための検索クエリを3つ生成してください。"},
                {"role": "user", "content": f"書籍「{book_title}」に関する書評、要約、内容解説を検索するための最適な検索クエリを3つ生成してください。"}
            ]
        )
        
        search_queries = response.choices[0].message.content.strip().split('\n')
        # 余分なフォーマットを削除
        cleaned_queries = [q.strip().replace('"', '').replace('- ', '') for q in search_queries if q.strip()]
        
        return {"search_queries": cleaned_queries[:3]}  # 最大3つのクエリを返す
    
    return query_optimizer_agent

# Web検索を実行するエージェント (User's provided logic)
def create_web_search_agent():
    # 1) 関数定義 (as per user's suggestion)
    functions = [
        {
            "name": "web_search",
            "description": "Web 検索を実行して結果を返す",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "検索クエリ"
                    }
                },
                "required": ["query"]
            }
        }
    ]
    
    def web_search_agent(state: WorkflowState) -> WorkflowState:
        book_title = state["book_title"]
        # For testing, we use a simplified query as per previous steps.
        # The query_optimizer_agent should provide the actual search_queries for production.
        # However, the user's example uses a single query directly.
        # We will adapt to process multiple queries if they come from query_optimizer.
        
        # Use the search_queries from the state, or a default simple one for testing
        # If query_optimizer is used, it should populate state["search_queries"]
        # For now, we'll stick to the simplified single query test to match the user's direct example structure
        # and previous debugging steps.
        search_query = f"{book_title} 書籍 レビュー"
        
        # 外部で定義したclientを使用
        # client = OpenAI()
        all_results = []
        
        try:
            print(f"Web検索実行中: '{search_query}'...")
            
            # Responses APIを使用してWeb検索
            web_search_response = client.responses.create(
                model="gpt-4o",
                input=f"以下の書籍に関する情報を詳しく検索してください: {search_query}",
                tools=[{"type": "web_search"}]
            )
            
            # レスポンス全体の構造をデバッグ表示
            print("Responses API レスポンス構造:")
            print(f"- レスポンスタイプ: {type(web_search_response)}")
            print(f"- 利用可能な属性: {dir(web_search_response)}")
            
            # テキスト内容を取得 (属性名により異なる可能性があるため複数試す)
            web_search_result_text = None
            
            # 方法1: output_textプロパティ
            if hasattr(web_search_response, 'output_text'):
                web_search_result_text = web_search_response.output_text
                print("output_textプロパティから検索結果を取得しました")
            
            # 方法2: outputリストのコンテンツ
            elif hasattr(web_search_response, 'output') and web_search_response.output:
                # outputがリストの場合
                if isinstance(web_search_response.output, list) and len(web_search_response.output) > 0:
                    output_item = web_search_response.output[0]
                    print(f"- outputの最初の要素のタイプ: {type(output_item)}")
                    print(f"- outputの最初の要素の属性: {dir(output_item)}")
                    
                    # contentプロパティを試す
                    if hasattr(output_item, 'content'):
                        web_search_result_text = output_item.content
                        print("output[0].contentから検索結果を取得しました")
                    # textプロパティを試す
                    elif hasattr(output_item, 'text'):
                        web_search_result_text = output_item.text
                        print("output[0].textから検索結果を取得しました")
                    # valueプロパティを試す
                    elif hasattr(output_item, 'value'):
                        web_search_result_text = output_item.value
                        print("output[0].valueから検索結果を取得しました")
                # outputが辞書の場合
                elif isinstance(web_search_response.output, dict):
                    web_search_result_text = str(web_search_response.output)
                    print("output辞書から検索結果を取得しました")
            
            # 最終手段: 文字列表現を使用
            if web_search_result_text is None:
                web_search_result_text = str(web_search_response)
                print("レスポンス全体の文字列表現を使用します")
            
            # 検索結果を保存
            all_results.append({
                "query": search_query,
                "content": f"Web検索結果: {search_query}\n\n{web_search_result_text[:1000]}...(省略)" 
                          if len(web_search_result_text) > 1000 else web_search_result_text,
                "raw_data": {
                    "web_search_response_content": web_search_result_text,
                    "response_type": str(type(web_search_response)),
                    "available_attrs": str(dir(web_search_response))
                }
            })
            
        except Exception as e:
            error_message = f"Web検索中にエラーが発生しました: {str(e)}"
            print(f"エラー: {error_message}")
            import traceback
            traceback_str = traceback.format_exc()
            print(traceback_str)
            
            all_results.append({
                "query": search_query,
                "content": error_message,
                "raw_data": {
                    "error": str(e),
                    "traceback": traceback_str
                }
            })
        
        return {"search_results": all_results}
    
    return web_search_agent

# 初期要約を作成するエージェント
def create_summary_agent():
    def summary_agent(state: WorkflowState) -> WorkflowState:
        book_title = state["book_title"]
        search_results = state["search_results"]
        
        # 検索結果をテキストとして連結
        search_text = ""
        for result in search_results:
            search_text += f"検索クエリ: {result['query']}\n"
            search_text += f"検索結果 (要約済み): {result['content']}\n\n"

            # raw_dataの内容も追加 (安全に取得)
            if result.get("raw_data") and isinstance(result["raw_data"], dict):
                # web_search_response_contentキーが存在する場合のみ処理
                raw_content = result["raw_data"].get("web_search_response_content")
                if raw_content:
                    # 文字列が長すぎる場合は切り詰める
                    if len(raw_content) > 10000:
                        shortened_content = raw_content[:10000] + "...(省略)"
                        search_text += f"検索結果 (生データ、一部省略):\n---\n{shortened_content}\n---\n\n"
                    else:
                        search_text += f"検索結果 (生データ):\n---\n{raw_content}\n---\n\n"
                
                # エラー情報がある場合はそれも追加
                if "error" in result["raw_data"]:
                    search_text += f"エラー情報: {result['raw_data']['error']}\n\n"
                
                # raw_dataに他のキーがあればそれも表示 (デバッグ用)
                other_keys = [k for k in result["raw_data"].keys() 
                              if k not in ["web_search_response_content", "error", "traceback"]]
                if other_keys:
                    other_data = {k: result["raw_data"][k] for k in other_keys}
                    search_text += f"その他のデータ: {json.dumps(other_data, ensure_ascii=False)}\n\n"

            search_text += "==========\n\n" # クエリごとの区切りを明確に

        print("--- 要約生成の入力テキスト (先頭1000文字) ---")
        print(search_text[:1000] + "... (省略)" if len(search_text) > 1000 else search_text)
        print("---------------------------------------------------------")

        try:
            # 検索結果から要約を生成
            response = client.chat.completions.create(
                model="gpt-4o", # Use gpt-4o for potentially better summary quality
                messages=[
                    {"role": "system", "content": f"""あなたは書籍要約の専門家です。以下の検索結果に基づいて、書籍「{book_title}」の要約を作成してください。
検索結果には「要約済み」と「生データ」が含まれています。両方を参考に、できるだけ正確で包括的な情報を盛り込んでください。

以下の構成でGitHubのMarkdown形式で書籍の要約を作成してください：

## 書籍情報
- タイトル: {book_title}
- 著者：（分かれば記入）
- 出版社：（分かれば記入）
- 出版年：（分かれば記入）

## 概要
（書籍の内容を200-300字程度で簡潔に要約）

## 主なポイント
- （ポイント1）
- （ポイント2）
- （ポイント3）
- （必要に応じて追加）

## 評価
（レビューや書評からわかる評価を要約）

## 参考文献
- 検索結果の「生データ」に含まれる情報源（URLなど）があれば記載してください。

注意事項:
1. バッククォート(```)でMarkdownコードブロックを作成しないでください
2. 見出しレベルは「##」から始め、「#」は使わないでください
3. 箇条書きは「- 」を使用してください
4. GitHubで正常に表示されるMarkdown記法を使用してください

必ず上記のフォーマットに従ってください。空欄にする項目があっても、見出しは削除しないでください。"""},
                    {"role": "user", "content": f"次の検索結果から書籍「{book_title}」の要約を作成してください:\n\n{search_text}"}
                ],
                max_tokens=4000
            )
            
            summary = response.choices[0].message.content
            
            # 要約が不完全な場合はプレースホルダーを用意
            if not summary or summary.strip() == "":
                summary = f"""
## 書籍情報
- タイトル: {book_title}
- 著者：（情報が取得できませんでした）
- 出版社：（情報が取得できませんでした）
- 出版年：（情報が取得できませんでした）

## 概要
書籍情報の取得中にエラーが発生しました。検索結果から適切な情報を抽出できませんでした。

## 主なポイント
- 情報が取得できませんでした
- 再度実行してみてください

## 評価
情報が取得できませんでした。

## 参考文献
- （情報が取得できませんでした）
"""
            
        except Exception as e:
            summary = f"""
## 書籍情報
- タイトル: {book_title}
- 著者：（情報が取得できませんでした）
- 出版社：（情報が取得できませんでした）
- 出版年：（情報が取得できませんでした）

## 概要
書籍情報の取得中にエラーが発生しました: {str(e)}

## 主なポイント
- エラーが発生したため情報が取得できませんでした
- 再度実行してみてください

## 評価
情報が取得できませんでした。

## 参考文献
- （情報が取得できませんでした）
"""
        
        return {"initial_summary": summary}
    
    return summary_agent

# フィードバックを与えるエージェント
def create_feedback_agent():
    prompt = ChatPromptTemplate.from_messages([
        ("system", """あなたは書籍レビューの品質を向上させる専門家です。
以下の書籍要約に対して、以下の観点からフィードバックを提供してください：
1. 情報の正確性と完全性
2. 構成と論理的な流れ
3. 重要なポイントの抽出
4. 客観性と公平性
5. 改善すべき点

フィードバックは具体的で建設的なものにしてください。"""),
        MessagesPlaceholder(variable_name="messages"),
    ])

    model = ChatOpenAI(model="gpt-4o")
    chain = prompt | model | StrOutputParser()
    
    def feedback_agent(state: WorkflowState) -> WorkflowState:
        messages = state["messages"]
        summary = state["initial_summary"]
        
        feedback = chain.invoke({
            "messages": messages + [HumanMessage(content=f"以下の書籍要約に対するフィードバックをお願いします：\n\n{summary}")]
        })
        
        return {"agent_feedback": feedback}

    return feedback_agent

# 推敲を行うエージェント
def create_revision_agent():
    prompt = ChatPromptTemplate.from_messages([
        ("system", """あなたは書籍レビューの推敲を担当する専門家です。
元の要約とフィードバックを参考に、より良い書籍要約を作成してください。
以下の点に注意してください：
1. フィードバックの指摘を反映する
2. 元の内容の本質は保持する
3. より明確で読みやすい文章にする
4. 書籍の価値や特徴を適切に伝える
5. GitHub互換のMarkdown形式を維持する

特に以下のMarkdown記法の注意点を守ってください：
- バッククォート(```)でコードブロックを作成しない
- 見出しレベルは「##」から始め、「#」は使わない
- 箇条書きは「- 」を使用する
- 複雑なフォーマットは避け、シンプルなMarkdownを使用する"""),
        MessagesPlaceholder(variable_name="messages"),
    ])

    model = ChatOpenAI(model="gpt-4o")
    chain = prompt | model | StrOutputParser()
    
    def revision_agent(state: WorkflowState) -> WorkflowState:
        messages = state["messages"]
        summary = state["initial_summary"]
        feedback = state["agent_feedback"]
        book_title = state["book_title"]
        
        revised_summary = chain.invoke({
            "messages": messages + [
                HumanMessage(content=f"書籍タイトル：{book_title}\n\n元の要約：\n{summary}\n\nフィードバック：\n{feedback}\n\n上記を参考に、GitHub対応のMarkdown形式でより良い書籍要約を作成してください。バッククォート(```)でコードブロックを作成せず、シンプルな形式を保ってください。")
            ]
        })
        
        return {"final_summary": revised_summary}

    return revision_agent

# 結果を保存するエージェント
def create_save_agent():
    def save_agent(state: WorkflowState) -> WorkflowState:
        today = datetime.now().strftime("%Y%m%d")
        book_title = state["book_title"]
        # ファイル名に使える文字のみを使用し、英数字と一部記号のみにする
        safe_title = "".join(c if c.isalnum() or c in ['_', '-'] else '_' for c in book_title)
        # スペースはアンダースコアに置換
        safe_title = safe_title.replace(' ', '_')
        filename = f"output/book_{safe_title}_{today}.md"
        
        # 出力ディレクトリが存在することを確認
        os.makedirs("output", exist_ok=True)
        
        # 検索結果から参考URLを抽出
        search_results = state["search_results"]
        reference_urls = []
        
        # 正規表現でURLを抽出する関数
        def extract_urls_from_text(text):
            if not text or not isinstance(text, str):
                return []
            # URLの正規表現パターン
            url_pattern = r'https?://[^\s\)\]\"\']+(?:\.[^\s\)\]\"\']+)+[^\s\)\]\"\'\.,]'
            return re.findall(url_pattern, text)
        
        # 検索結果からURLを抽出
        for result in search_results:
            if result.get("raw_data") and isinstance(result["raw_data"], dict):
                raw_content = result["raw_data"].get("web_search_response_content")
                if raw_content:
                    urls = extract_urls_from_text(raw_content)
                    if urls:
                        for url in urls:
                            if url not in reference_urls:
                                reference_urls.append(url)
        
        # GitHub/Markdownで崩れないように整形した内容をファイルに書き込む
        with open(filename, "w", encoding="utf-8") as f:
            # タイトル
            f.write(f"# 「{book_title}」書籍要約 ({today})\n\n")
            
            # 初期要約
            f.write("## 初期要約\n\n")
            f.write(state["initial_summary"])
            
            # フィードバック
            f.write("\n\n## フィードバック\n\n")
            f.write(state["agent_feedback"])
            
            # 改善後の要約
            f.write("\n\n## 改善後の要約\n\n")
            f.write(state["final_summary"])
            
            # 参考URLがあれば記載
            if reference_urls:
                f.write("\n\n## 参考URL\n\n")
                for url in reference_urls:
                    f.write(f"- {url}\n")
            
            # 検索クエリと結果の詳細（デバッグ情報）
            f.write("\n\n## 検索クエリと結果 (デバッグ情報)\n\n")
            for i, result in enumerate(search_results, 1):
                # 検索クエリ
                f.write(f"### 検索クエリ {i}: {result['query']}\n\n")
                
                # 要約済み応答を表示（コードブロック内に入れる）
                content = result.get('content', 'N/A')
                # GitHubでの表示時にネストしたコードブロックが問題を起こさないよう修正
                content = content.replace("```", "~~~")
                f.write(f"#### 要約済み応答:\n\n{content}\n\n")
                
                # 生データは詳細情報を省略
                f.write("#### 生データ (raw_data):\n\n")
                f.write("生データの詳細は省略します。\n\n")
                
        print(f"結果は「{filename}」に保存されました。")
        return state
    return save_agent

# ワークフローの構築
def create_workflow():
    # エージェントの作成
    query_optimizer = create_query_optimizer_agent()
    web_search = create_web_search_agent()
    summary_agent = create_summary_agent()
    feedback_agent = create_feedback_agent()
    revision_agent = create_revision_agent()
    save_agent = create_save_agent()
    
    # グラフの構築
    workflow = StateGraph(WorkflowState)
    
    # ノードの追加
    workflow.add_node("query_optimizer", query_optimizer)
    workflow.add_node("web_search", web_search)
    workflow.add_node("summary", summary_agent)
    workflow.add_node("feedback", feedback_agent)
    workflow.add_node("revision", revision_agent)
    workflow.add_node("save", save_agent)
    
    # エッジの設定
    workflow.add_edge("query_optimizer", "web_search")
    workflow.add_edge("web_search", "summary")
    workflow.add_edge("summary", "feedback")
    workflow.add_edge("feedback", "revision")
    workflow.add_edge("revision", "save")
    
    # エントリーポイントと終了ポイントの設定
    workflow.set_entry_point("query_optimizer")
    workflow.set_finish_point("save")
    
    return workflow.compile()

# メインの実行関数
async def run_workflow(book_title: str) -> dict:
    # ワークフローの作成
    workflow = create_workflow()
    
    # 初期状態の設定
    initial_state = {
        "messages": [],
        "book_title": book_title,
        "search_results": [],
        "initial_summary": "",
        "agent_feedback": "",
        "final_summary": ""
    }
    
    # ワークフローの実行
    result = await workflow.ainvoke(initial_state)
    
    return result 