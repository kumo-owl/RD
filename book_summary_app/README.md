# 書籍要約アプリケーション

このアプリケーションは、書籍のタイトルを入力として、OpenAIのWeb検索機能を使用して書籍に関する情報を検索し、要約を生成します。生成された要約は、AI によるフィードバックと推敲のプロセスを経て、最終的な要約として出力されます。

## 特徴

- OpenAIのResponses APIを使用したWeb検索機能
- LangGraphを活用したエージェントベースのワークフロー
- 初期要約、フィードバック、および改善された最終要約の生成
- Markdown形式での要約出力

## 必要条件

- Python 3.9以上
- OpenAI API キー
- 以下のPythonパッケージ:
  - openai
  - langgraph
  - langchain
  - langchain-openai
  - python-dotenv

## インストール

1. リポジトリをクローンするか、ダウンロードします。

```bash
git clone <repository-url>
cd book_summary_app
```

2. 必要なパッケージをインストールします。

```bash
pip install -r requirements.txt
```

3. OpenAI API キーを設定します。

**方法1: .envファイルを使用する (推奨)**

プロジェクトのルートディレクトリに`.env`ファイルを作成し、以下のように記述します:

```
OPENAI_API_KEY=your-api-key-here
```

例として`.env.example`ファイルが提供されています。これをコピーして使用できます:

```bash
cp .env.example .env
```

その後、テキストエディタで`.env`ファイルを開き、実際のAPIキーを入力してください。

**方法2: 環境変数を直接設定する**

```bash
# Linuxまたは macOS
export OPENAI_API_KEY='your-api-key'

# Windowsの場合
set OPENAI_API_KEY=your-api-key
```

## 使い方

### コマンドラインから実行する

書籍のタイトルを引数として渡して実行します：

```bash
python src/book_summary.py "ハリーポッター 賢者の石"
```

引数なしで実行すると、対話的に書籍タイトルを入力するよう促されます：

```bash
python src/book_summary.py
```

### 出力

生成された要約は `output/` ディレクトリに保存され、次の情報が含まれます：

- 初期要約
- AIによるフィードバック
- 改善された最終要約
- デバッグ情報（検索クエリと結果）

## ディレクトリ構造

```
book_summary_app/
├── README.md               # このファイル
├── requirements.txt        # 必要なパッケージリスト
├── .env.example            # 環境変数設定例
├── .env                    # 実際の環境変数設定（gitignoreに追加すべき）
├── src/                    # ソースコード
│   ├── book_summary.py     # メインアプリケーション
│   └── web_search_workflow.py # ワークフロー定義
├── examples/               # 使用例
└── output/                 # 生成された要約の保存先
```

## 仕組み

1. **クエリ最適化**: 書籍タイトルに基づいて最適な検索クエリを生成します。
2. **Web検索**: OpenAIのResponses APIを使用して、生成されたクエリで情報を検索します。
3. **初期要約**: Web検索の結果から書籍の初期要約を生成します。
4. **フィードバック**: 初期要約の品質を評価し、改善すべき点を指摘します。
5. **改訂**: フィードバックに基づいて要約を改善します。
6. **保存**: 最終的な要約をMarkdownファイルとして保存します。

## 注意事項

- このアプリケーションはOpenAI APIの使用に伴う料金が発生します。
- Web検索結果は完全に正確であるとは限りません。
- API制限により、大量のリクエストを短時間に行うとエラーが発生する可能性があります。
- `.env`ファイルには機密情報（APIキー）が含まれるため、Gitリポジトリにコミットしないでください。

## ライセンス

MITライセンス 