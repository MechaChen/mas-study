import os

import requests
import dotenv

dotenv.load_dotenv()

# with ... as 是 Python 的上下文管理器（Context Manager) 語法，離開 with 時會自動關閉連線、釋放資源
with requests.request(
  "POST",
  "https://api.deepseek.com/chat/completions",
  headers={
    "Content-Type": "application/json",
    "Accept": "application/json",
    "Authorization": f"Bearer {os.getenv('DEEPSEEK_API_KEY')}"
  },
  json={
    "model": "deepseek-reasoner",
    "messages": [
      {"role": "user", "content": "你好，你是？"}
    ],
    "stream": True
  }
) as resp:
  for line in resp.iter_lines(decode_unicode=True):
    if line: # 避免收到空行
      if line.startswith("data: "):
        data = line.lstrip("data:").strip()
        print("data:", data)
