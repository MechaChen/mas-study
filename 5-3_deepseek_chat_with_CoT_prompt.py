import os
import json
from openai import OpenAI
import dotenv
from openai.types.chat.chat_completion_chunk import ChoiceDeltaToolCall

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
    self.model="deepseek-chat"
    self.client = OpenAI()
    self.messages = [
      {
        "role": "system",
        "content": """
        你是一個擅長推理的智慧助手，你會先推理，並遵循以下格式
        1. 在 <think> 標籤內，詳細展示你的推理過程，將推理過程拆解成多個小步驟，並逐步執行，並在適當的過程中使用工具來幫你解決問題
        2. 在 <answer> 標籤內，回答最終 & 清晰的答案
        
        確保你的回答中不包含 `<think>` 或 `<answer>` 標籤之外的任何多餘文字
        """
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
          "description": "可以執行程式可執行的數學計算式，並回傳計算結果",
          "parameters": {
            "type": "object",
            "properties": {
              "expression": {
                "type": "string",
                "description": "程式可執行的數學計算式，例如 123 + 456 * 789"
              }
            }
          }
        }
      }
    ]

  def process_query(self, query: str) -> str:
    # 1. add user query to message history, and ask for DeepSeek streaming response
    # 2. collect the streaming response, including text & tool calls stream
    # 3. add merged text streams to message history
    # 4. if having tool calls, call the tool and get the result
    # 5. bring tool result to message history and ask DeepSeek again
    # 5. print the final DeepSeek response

    first_response = None
    is_tool_call = False
    merged_content = ""
    tool_calls_dict: dict[str, ChoiceDeltaToolCall] = {}
    tool_calls_json = []

    def ask_streaming_response():
      nonlocal first_response

      self.messages.append({
        "role": "user",
        "content": query,
      })
      print(f'\nAssistant: ', end='', flush=True)

      first_response = self.client.chat.completions.create(
        model=self.model,
        messages=self.messages,
        tools=self.available_tools,
        stream=True,
      )


    def merge_streaming_response():
      nonlocal first_response, is_tool_call, merged_content, tool_calls_dict, tool_calls_json

      for chunk in first_response:
        chunk_content = chunk.choices[0].delta.content
        chunk_tool_calls = chunk.choices[0].delta.tool_calls

        if chunk_tool_calls:
          is_tool_call = True

        if chunk_content:
          merged_content += chunk_content
          print(chunk_content, end='', flush=True)

        if chunk_tool_calls:
          for chunk_tool_call in chunk_tool_calls:
            if chunk_tool_call.index not in tool_calls_dict:
              tool_calls_dict[chunk_tool_call.index] = chunk_tool_call
            else:
              tool_calls_dict[chunk_tool_call.index].function.arguments += chunk_tool_call.function.arguments
      
      tool_calls_json = [tool_call for tool_call in tool_calls_dict.values()]


    def add_merged_content_history():
      self.messages.append({
        "role": "assistant",
        "content": merged_content if merged_content else None,
        "tool_calls": tool_calls_json if tool_calls_json else None
      })


    def call_tools_and_append_results_history():
      for tool_call in tool_calls_json:
        tool_name = tool_call.function.name
        tool_args = json.loads(tool_call.function.arguments)
        print(f'\nTool name: {tool_name}')
        print(f'Tool arguments: {tool_args}')

        tool_result = self.tools[tool_name](**tool_args)
        print(f'Tool result: {tool_result}')

        self.messages.append({
          'tool_call_id': tool_call.id,
          "role": "tool",
          "name": tool_name,
          "content": tool_result
        })


    def ask_streaming_summary():
      second_response = self.client.chat.completions.create(
        model=self.model,
        messages=self.messages,
        tools=self.available_tools,
        tool_choice="none",
        stream=True,
      )

      for chunk in second_response:
        chunk_content = chunk.choices[0].delta.content

        if chunk_content:
          print(chunk_content, end='', flush=True)


    ask_streaming_response()
    merge_streaming_response()
    add_merged_content_history()

    if is_tool_call:
      call_tools_and_append_results_history()
      ask_streaming_summary()
  
    

  def chat_loop(self):
    while True:
      try:
        print('\n\n================')
        query = input('Query: ')
        
        if query.lower() == 'quit':
          break
        self.process_query(query)

      except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
  agent = ReActAgent()
  agent.chat_loop()