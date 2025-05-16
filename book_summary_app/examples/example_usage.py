#!/usr/bin/env python
"""
書籍要約アプリケーションの使用例

このスクリプトは、book_summaryモジュールを使用して
書籍の要約を生成する方法を示しています。
"""

import asyncio
import sys
import os
from pathlib import Path

# 親ディレクトリのsrcをモジュールパスに追加
current_dir = Path(__file__).resolve().parent
parent_dir = current_dir.parent
sys.path.append(str(parent_dir / "src"))

# dotenvがインストールされていれば、.envファイルから環境変数を読み込む
try:
    from dotenv import load_dotenv
    env_file = parent_dir / ".env"
    if env_file.exists():
        load_dotenv(env_file)
        print(f".envファイルを読み込みました: {env_file}")
except ImportError:
    print("python-dotenvがインストールされていません。環境変数を直接設定してください。")

from book_summary import main

async def run_examples():
    """いくつかの書籍タイトルでアプリケーションを実行する例"""
    
    # 例1: シンプルなタイトル
    print("=== 例1: シンプルなタイトル ===")
    await main("1984")
    
    # 例2: 著者名を含むタイトル
    print("\n\n=== 例2: 著者名を含むタイトル ===")
    await main("村上春樹 海辺のカフカ")
    
    # 例3: シリーズ名を含むタイトル
    print("\n\n=== 例3: シリーズ名を含むタイトル ===")
    await main("ハリーポッター 賢者の石")

if __name__ == "__main__":
    # OpenAI API キーが設定されているか確認
    if not os.environ.get("OPENAI_API_KEY"):
        print("エラー: 環境変数 'OPENAI_API_KEY' が設定されていません。")
        print("OpenAI API キーを設定してから再実行してください。")
        sys.exit(1)
    
    asyncio.run(run_examples()) 