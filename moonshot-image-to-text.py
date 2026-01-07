import base64

import dotenv
import os
from openai import OpenAI

dotenv.load_dotenv()

client = OpenAI(
  base_url="https://api.moonshot.ai/v1",
  api_key=os.getenv('MOONSHOT_API_KEY')
)

image_path = 'images/japan_tokyo.jpeg'

with open(image_path, "rb") as f:
  image_data = f.read()

# 使用 python 標準的 base64.b64encode 函數將圖片編碼成 base64 字串
image_url = f"data:image/jpeg;base64,{base64.b64encode(image_data).decode('utf-8')}"

response = client.chat.completions.create(
  model="moonshot-v1-8k-vision-preview",
  messages=[
    {
      "role": "user",
      "content": [{"type": "image_url", "image_url": {"url": image_url}}]
    }
  ]
)

print(f"圖片描述：{response.choices[0].message.content}")