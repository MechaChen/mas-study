import os
import json
from openai import OpenAI
import dotenv

dotenv.load_dotenv()

def calculator(expression: str) -> float:
  try:
    result = eval(expression)
    return json.dumps({"result": result})
  except Exception as e:
    return f"Expression Error: {str(e)}"

class ReActAgent:
  def __init__(self):
    # 1. initialize OpenAI SDK
    # 2. initialize system prompt & messages history
    # 3. initialize available tools
    self.model="deepseek-reasoner"
    self.client = OpenAI()
    self.messages = [
      {
        "role": "system",
        "content": "你是一個智慧助手，你會優先使用工具來幫你解決問題 & 回答問題，另外如果不知道就說無法回答"
      }
    ]
    self.tools = {
      "calculator": calculator
    }
    self.available_tools = [
      {
        "type": "function",
        "function": {
          "name": "calculator",
          "description": "可以執行數學系算式並回傳結果",
          "parameters": {
            "type": "object",
            "properties": {
              "expression": {
                "type": "string",
                "description": "數學算式，例如 123 + 456 * 789"
              }
            }
          }
        }
      }
    ]

  def process_query(self, query: str) -> str:
    # 1. add user message to message history
    # 2. feed message to LLM
    # 3. according to LLM response, decide whether to use tool or not
    self.messages.append({
      "role": "user",
      "content": query,
    })

    response = self.client.chat.completions.create(
      model=self.model,
      messages=self.messages,
      tools=self.available_tools
    )

    message = response.choices[0].message
    self.messages.append(message.model_dump())
    
    tool_calls = message.tool_calls

    if tool_calls:
      for tool_call in tool_calls:
        print(f"Tool name: {tool_call.function.name}")

        tool_name = tool_call.function.name
        tool_args = json.loads(tool_call.function.arguments)

        tool_result = self.tools[tool_name](**tool_args)
        print(f"Tool result: {tool_result}")

        self.messages.append({
          "tool_call_id": tool_call.id,
          "role": "tool",
          "name": tool_name,
          "content": tool_result
        })

        second_response = self.client.chat.completions.create(
          model=self.model,
          messages=self.messages,
          tools=self.available_tools,
          tool_choice="none",
        )

        second_response_message = second_response.choices[0].message
        self.messages.append(second_response_message.model_dump())
        return f'Assistant: {second_response_message.content}'
    else:
      return f"Asssitant: {message.content}" 

  def chat_loop(self):
    while True:
      try:
        query = input('\nQuery: ')
        
        if query.lower() == 'quit':
          break
        response = self.process_query(query)
        print(response)

      except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
  agent = ReActAgent()
  agent.chat_loop()