import base64
import os

import dotenv
import requests

dotenv.load_dotenv()

image_path = 'images/japan_tokyo.jpeg'

with open(image_path, "rb") as f:
  image_data = f.read()

# 使用 python 標準的 base64.b64encode 函數將圖片編碼成 base64 字串
image_url = f"data:image/jpeg;base64,{base64.b64encode(image_data).decode('utf-8')}"

response = requests.request(
  "POST",
  "https://api.moonshot.ai/v1/chat/completions",
  headers={
    "Content-Type": "application/json",
    "Accept": "application/json",
    "Authorization": f"Bearer {os.getenv('MOONSHOT_API_KEY')}"
  },
  json={
    "model": "moonshot-v1-8k-vision-preview",
    "messages": [
      {
        "role": "user",
        "content": [
          {
            "type": "image_url",
            "image_url": {
              "url": image_url
            }
          },
          {
            "type": "text",
            "text": "請描述這個圖片"
          }
        ]
      }
    ]
  }
)

print(response.json())