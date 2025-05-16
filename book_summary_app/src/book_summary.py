#!/usr/bin/env python
"""
書籍要約アプリケーション

書籍のタイトルを入力として、OpenAI APIを使用してWeb検索を行い、
書籍の要約を生成します。生成された要約は、フィードバックと推敲のプロセスを経て
最終的な要約として出力されます。
"""

import asyncio
import sys
import os
from pathlib import Path
import argparse

# .envファイルから環境変数を読み込む
try:
    from dotenv import load_dotenv
    # プロジェクトのルートディレクトリの.envファイルを読み込む
    root_dir = Path(__file__).resolve().parent.parent
    env_path = root_dir / ".env"
    if env_path.exists():
        load_dotenv(env_path)
        print(f".envファイルを読み込みました: {env_path}")
    else:
        print(f"警告: .envファイルが見つかりません: {env_path}")
except ImportError:
    print("警告: python-dotenvがインストールされていません。pip install python-dotenvでインストールしてください。")

# src ディレクトリを追加してモジュールを読み込めるようにする
src_dir = Path(__file__).resolve().parent
sys.path.append(str(src_dir))

from web_search_workflow import run_workflow

def check_openai_api_key():
    """OpenAI API Keyが設定されているか確認する"""
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("エラー: 環境変数 'OPENAI_API_KEY' が設定されていません。")
        print("以下のいずれかの方法でAPIキーを設定してください：")
        print("1. .envファイルにOPENAI_API_KEY=your-api-keyを追加")
        print("2. export OPENAI_API_KEY='your-api-key'で環境変数を設定")
        return False
    return True

async def main(book_title=None):
    """メイン実行関数"""
    # API キーが設定されているか確認
    if not check_openai_api_key():
        return
    
    # 書籍タイトルを取得
    if not book_title:
        book_title = input("要約する書籍のタイトルを入力してください: ")
    
    if not book_title:
        print("書籍タイトルが入力されていません。")
        return

    try:
        # output ディレクトリを作成
        output_dir = Path(src_dir).parent / "output"
        output_dir.mkdir(exist_ok=True)
        
        print(f"「{book_title}」の情報をWeb検索し、要約を作成します...")
        result = await run_workflow(book_title)
        
        print("\n=== 処理が完了しました ===")
        
    except Exception as e:
        print(f"エラーが発生しました: {str(e)}")
        import traceback
        print(traceback.format_exc())

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="書籍要約アプリケーション")
    parser.add_argument("book_title", nargs="?", help="要約する書籍のタイトル")
    args = parser.parse_args()
    
    asyncio.run(main(args.book_title)) 