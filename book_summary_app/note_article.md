# 書籍要約を自動化！LangGraphとOpenAIを使った書籍要約アプリの作り方

![アイキャッチ画像：本とAIのイメージ](https://images.unsplash.com/photo-1532012197267-da84d127e765?ixlib=rb-4.0.3)

こんにちは！今回は、**OpenAI API**と**LangGraph**を使って、書籍の要約を自動生成するアプリケーションを作る方法を紹介します。AIの力を借りて、本の内容を素早く把握したいと思ったことはありませんか？このアプリケーションを使えば、書籍のタイトルを入力するだけで、その本の要約を自動的に生成できます！

## このアプリケーションでできること

- 書籍タイトルを入力するだけで要約を自動生成
- Web検索でリアルタイムの情報を収集
- AIによるフィードバックと改善で高品質な要約を作成
- 要約結果をMarkdownファイルとして保存

実際に生成された要約例はこちら👇

![要約例のイメージ](https://images.unsplash.com/photo-1501504905252-473c47e087f8?ixlib=rb-4.0.3)

## 仕組み：エージェントベースの設計

このアプリケーションは**LangGraph**というライブラリを使って、複数のAIエージェントが連携する仕組みになっています。各エージェントは特定の役割を担当し、以下のような流れで動作します：

1. **クエリ最適化エージェント**：書籍タイトルから最適な検索クエリを生成
2. **Web検索エージェント**：OpenAIのWeb検索機能を使って情報を収集
3. **要約エージェント**：検索結果から初期要約を作成
4. **フィードバックエージェント**：要約の品質を評価し改善点を提案
5. **改訂エージェント**：フィードバックを基に要約を改善
6. **保存エージェント**：最終的な要約をMarkdownファイルとして保存

この設計により、単純な要約よりも高品質な結果が得られます。特に、フィードバックと改訂のプロセスが入ることで、より読みやすく正確な要約になります。

## 実装のポイント

### 1. エージェントベースの設計

LangGraphを使うことで、複数のエージェントが協力して作業する仕組みを簡単に実装できます。各エージェントは特定のタスクに特化しているため、全体のクオリティが向上します。

```python
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
    # ... その他のノード
    
    # エッジの設定
    workflow.add_edge("query_optimizer", "web_search")
    workflow.add_edge("web_search", "summary")
    # ... その他のエッジ
    
    return workflow.compile()
```

### 2. OpenAIのResponses APIの活用

最新のOpenAI APIを使うことで、Web検索機能を簡単に実装できます。

```python
# Web検索の実行
web_search_response = client.responses.create(
    model="gpt-4o",
    input=f"以下の書籍に関する情報を詳しく検索してください: {search_query}",
    tools=[{"type": "web_search"}]
)
```

### 3. フィードバックループの実装

初期要約に対してAIがフィードバックを行い、それを基に要約を改善するループを実装しています。これにより要約の品質が大幅に向上します。

```python
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
    # ... 以下省略
```

## 実際の使い方

このアプリケーションの使い方はとても簡単です！

1. リポジトリをクローンする
2. 必要なパッケージをインストールする
3. OpenAI APIキーを設定する
4. 以下のコマンドを実行するだけ！

```bash
python src/book_summary.py "ハリーポッター 賢者の石"
```

引数なしで実行すると、対話的に書籍タイトルを入力することもできます：

```bash
python src/book_summary.py
```

## 工夫したポイント

### 1. エラーハンドリングの強化

Web検索や要約生成中にエラーが発生しても、適切に処理して最後まで実行できるようにしています。

### 2. 出力フォーマットの改善

GitHubなどのMarkdownビューアでも正しく表示されるように、出力フォーマットを最適化しています。特にネストされたコードブロックの扱いに注意しました。

### 3. 適切なプロンプト設計

各エージェントのプロンプトを工夫することで、より質の高い出力を得られるようにしています。特に、要約エージェントには詳細な指示を与えて構造化された要約を生成させています。

## 今後の改善点

- 複数の書籍を比較する機能
- 要約の長さを指定できるオプション
- 特定のトピックに焦点を当てた要約機能
- 多言語対応（英語や他の言語の書籍も要約できるように）

## おわりに

このプロジェクトを通じて、LangChainとLangGraphというフレームワークの可能性を実感しました。複数のAIエージェントを連携させることで、単一のプロンプトでは難しい複雑なタスクも実現できます。

みなさんもぜひ、このアプリケーションを試してみてください！カスタマイズや機能追加も簡単にできるので、自分のニーズに合わせて改良してみるのも楽しいと思います。

GitHubリポジトリ: [https://github.com/kumo-owl/RD](https://github.com/kumo-owl/RD)

---

この記事が参考になれば嬉しいです。何か質問や感想があれば、コメントで教えてください！ 