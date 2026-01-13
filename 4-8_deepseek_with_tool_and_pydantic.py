import dotenv
from openai import OpenAI
from pydantic import BaseModel, Field, EmailStr

dotenv.load_dotenv()

class UserInfo(BaseModel):
  """
  UserInfo 模型，用來儲存用戶的年齡、姓名、信箱
  """
  name: str = Field(..., description="用戶名稱")
  age: int = Field(..., description="用戶年齡")
  email: EmailStr = Field(..., description="用戶信箱")


client = OpenAI()

response = client.chat.completions.create(
  model="deepseek-chat",
  messages=[
    {
      "role": "user",
      "content": "用戶名稱：Benson Chen，用戶年齡： 32 歲，用戶信箱： benson@example.com"
    }
  ],
  tools=[
    {
      "type": "function",
      "function": {
        "name": UserInfo.__name__,
        "description": UserInfo.__doc__,
        "parameters": UserInfo.model_json_schema()
      }
    }
  ],
  tool_choice={
    "type": "function",
    "function": {
      "name": UserInfo.__name__,
    }
  }
)

tool_args = response.choices[0].message.tool_calls[0].function.arguments
user_info = UserInfo.model_validate_json(tool_args)

print(f"User info: {user_info}")