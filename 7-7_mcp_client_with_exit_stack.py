import asyncio
from contextlib import AsyncExitStack

from mcp import StdioServerParameters, ClientSession
from mcp.client.stdio import stdio_client

async def main() -> None:
    # 1. 準備 STDIO MCP server 的呼叫參數
    server_params = StdioServerParameters(
        command="uv",
        args=[
            "--directory",
            "/Users/claire/Desktop/Benson/mas-study",
            "run",
            "7-6_mcp_server_demo.py" # MCP server 的主程式碼
        ]
    )

    # 2. 初始化一個異步上下文管理器
    exit_stack = AsyncExitStack()

    try:
        # 3. 建立 STDIO 的客戶端
        transport = await exit_stack.enter_async_context(stdio_client(server_params))
        # 4. 獲取寫入和寫出流
        stdio, write = transport

        # 5. 創建客戶端會話上下文
        session = await exit_stack.enter_async_context(ClientSession(stdio, write))

        # 6. 初始化 MCP server
        await session.initialize()

        # 7. 獲取工具列表訊息
        list_tools_response = await session.list_tools()
        tools = list_tools_response.tools
        print('工具列表:', [tool.name for tool in tools])

        # 8. 調用指定的工具
        call_tool_response = await session.call_tool('calculator', {"expression": "564*34+12.4/455**2"})
        print("工具結果:", call_tool_response)
    finally:
        await exit_stack.aclose()

if __name__ == "__main__":
    asyncio.run(main())            

