import json
import requests
import os
import dotenv

from mcp.server import MCPServer

mcp = MCPServer(name="三方 API")

# 應用配置訊息
DIFY_API = "https://api.dify.ai/v1/chat-messages"

dotenv.load_dotenv()
dify_api_key = os.getenv("DIFY_API_KEY")

@mcp.tool()
async def call_dify_agent(query: str) -> str:
    """調用外部 Agent 實現對 query 的回答，這個 Agent 支持天氣查詢、網路內容搜索、獲取檔前時間等

    Args:
        query (str): 用戶需要提問的問題

    Returns:
        str: 外部 Agent 對 query 生成的最終答案
    """
    answer = ""
    with requests.request(
        "POST",
        DIFY_API,
        headers={
            "Content-Type": "application/json",
            "Accept": "application/json",
            "Authorization": f"Bearer {dify_api_key}",
        },
        json={
            "inputs": {},
            "query": query,
            "response_mode": "streaming",
            "user": "mcp-client"
        }
    ) as resp:
        for line in resp.iter_lines(decode_unicode=True):
            if line:
                if line.startswith("data:"):
                    data = line.lstrip("data:").strip()
                    resp_obj = json.loads(data)

                    if resp_obj.get("event") in ("message", "agent_message"):
                        answer += resp_obj.get("answer", "")

    return answer.strip()

if __name__ == "__main__":
    mcp.run(transport="streamable-http", port=8001)

