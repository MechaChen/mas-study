import json
import os
import requests
from datetime import date
from typing import Literal

import dotenv
from openai import OpenAI
from openai.types.chat.chat_completion_chunk import ChoiceDeltaToolCall
from pydantic import BaseModel, Field

dotenv.load_dotenv()

# 定義 Google Map 服務基礎地址
GOOGLE_MAP_URL = f"https://mapstools.googleapis.com/mcp"
GOOGLE_MAP_API_KEY = os.getenv('GOOGLE_MAP_API_KEY') or os.getenv('GOOGLE_MAP_KEY')
if not GOOGLE_MAP_API_KEY:
  raise RuntimeError("請在 .env 設定 GOOGLE_MAP_API_KEY 或 GOOGLE_MAP_KEY")

SYSTEM_PROMPT = "你是一個強大的聊天機器人，請根據用戶的提問進行回覆，如果需要調用工具請直接調用，不知道請直接回覆不知道"


class ReActAgent:
  def __init__(self):
    self.model = "deepseek-chat"
    self.client = OpenAI()
    self.messages = [
      {
        "role": "system",
        "content": SYSTEM_PROMPT
      }
    ]
    self.available_tools = []
    self.headers = {
      "X-Goog-Api-Key": GOOGLE_MAP_API_KEY,
      "Content-Type": "application/json",
      "Accept": "application/json, text/event-stream"
    }
    self.init_google_map_mcp()

  def init_google_map_mcp(self) -> None:
    """初始化 Google Map MCP 服務"""
    # 1. 調用 Google Map MCP 服務獲取工具訊息
    tool_list_response = requests.post(
      GOOGLE_MAP_URL,
      headers = self.headers,
      json={
        "jsonrpc": '2.0',
        "id": 1,
        "method": 'tools/list',
        "params": {},
      }
    )
    tool_list_result = self.get_mcp_result(tool_list_response) # 取得結果

    # 2. 將每個 tool 拆解資料，放到 self 的 available tools 中
    self.available_tools = [{
      "type": "function",
      "function": {
        "name": tool["name"],
        "description": tool["description"],
        "parameters": tool["inputSchema"]
      }
    } for tool in tool_list_result["tools"]]

  @staticmethod
  def get_mcp_result(response: requests.Response) -> dict:
    """檢查 JSON-RPC 錯誤，避免直接存取不存在的 result。"""
    if not response.ok:
      detail = response.text
      if GOOGLE_MAP_API_KEY:
        detail = detail.replace(GOOGLE_MAP_API_KEY, "[REDACTED]")
      raise RuntimeError(f"Google Maps MCP HTTP {response.status_code}: {detail}")
    data = response.json()
    if not isinstance(data, dict):
      raise RuntimeError("Google Maps MCP returned an invalid JSON-RPC response")
    if "error" in data:
      raise RuntimeError(
        f"Google Maps MCP error: {json.dumps(data['error'], ensure_ascii=False)}"
      )
    if not isinstance(data.get("result"), dict):
      raise RuntimeError("Google Maps MCP response is missing a valid 'result'")
    return data["result"]

  @staticmethod
  def normalize_search_arguments(arguments: dict) -> dict:
    """使用 Postman 已驗證的 JSON 欄位格式，保留實際搜尋內容。"""
    aliases = {
      "text_query": "textQuery",
      "language_code": "languageCode",
      "region_code": "regionCode",
      "page_size": "pageSize",
      "include_extended_details": "includeExtendedDetails",
      "location_bias": "locationBias",
      "radius_meters": "radiusMeters",
    }

    def convert(value):
      if isinstance(value, dict):
        return {aliases.get(key, key): convert(item) for key, item in value.items()}
      if isinstance(value, list):
        return [convert(item) for item in value]
      return value

    normalized = convert(arguments)
    # 暫時使用與成功的 Postman 請求相同的結果數量上限。
    normalized["pageSize"] = min(normalized.get("pageSize", 3), 3)
    return normalized

  def call_google_map_mcp(self, name: str, arguments: dict) -> str:
    """調用 Google Map MCP 各項工具服務"""
    if name == "search_places":
      arguments = self.normalize_search_arguments(arguments)
    print("\nMCP Request Arguments:", json.dumps(arguments, ensure_ascii=False))
    # 1. 發起 POST 請求調用 Google Map MCP 服務
    tools_call_response = requests.post(GOOGLE_MAP_URL, headers=self.headers, json={
      "jsonrpc": "2.0",
      "id": 1,
      "method": "tools/call",
      "params": {
        "name": name,
        "arguments": arguments,
      }
    })
    tools_call_result = self.get_mcp_result(tools_call_response)

    # 2. 回傳內容
    return tools_call_result["content"][0]["text"]

  def process_query(self, query: str = '') -> None:
    if query:
      self.messages.append({
        "role": "user",
        "content": query,
      })

    response = self.client.chat.completions.create(
      model=self.model,
      messages=self.messages,
      tools=self.available_tools,
      stream=True,
    )

    merged_content = ""
    merged_tool_calls_map: dict[str, ChoiceDeltaToolCall] = {}
    is_tool_call = False

    for chunk in response:
      chunk_content = chunk.choices[0].delta.content
      chunk_tool_calls = chunk.choices[0].delta.tool_calls

      if chunk_content:
        merged_content += chunk_content
        print(chunk_content, end='', flush=True)

      if chunk_tool_calls:
        is_tool_call = True

        for chunk_tool_call in chunk_tool_calls:
          if chunk_tool_call.index not in merged_tool_calls_map:
            merged_tool_calls_map[chunk_tool_call.index] = chunk_tool_call
          else:
            merged_tool_calls_map[chunk_tool_call.index].function.arguments += chunk_tool_call.function.arguments


    merged_tool_calls = [tool_call for tool_call in merged_tool_calls_map.values()]

    self.messages.append({
      "role": "assistant",
      "content": merged_content if merged_content else None,
      "tool_calls": merged_tool_calls if merged_tool_calls else None
    })

    if is_tool_call:
      for tool_call in merged_tool_calls:
        tool_name = tool_call.function.name
        tool_arguments = json.loads(tool_call.function.arguments)
        print("\nTool Call: ", tool_name)
        print("\nTool Parameters: ", tool_arguments)

        # 調用工具
        try:
          result = self.call_google_map_mcp(tool_name, tool_arguments)
        except Exception as e:
          result = f"工具執行出錯，Error: {str(e)}"
          print(result)

        self.messages.append({
          "tool_call_id": tool_call.id,
          "role": "tool",
          "name": tool_name,
          "content": result
        })
      
      self.process_query()



  def chat_loop(self) -> None:
    while True:
      try:
        query = input('\nQuery: ')

        if query.lower() == 'quit':
          break

        self.process_query(query)

      except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
  agent = ReActAgent()
  agent.chat_loop()
