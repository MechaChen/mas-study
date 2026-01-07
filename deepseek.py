import dotenv
from openai import OpenAI

dotenv.load_dotenv()

client = OpenAI()

response = client.chat.completions.create(
    model="deepseek-reasoner",
    messages=[
        {"role": "user", "content": "你好，你是？"}
    ]
)

print(f"推理過程：{response.choices[0].message.reasoning_content}")
print(f"最終結果：{response.choices[0].message.content}")

# def main():
#     print("Hello from mas-study!")


# if __name__ == "__main__":
#     main()
