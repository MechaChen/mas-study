import json

from mcp.server import MCPServer

mcp = MCPServer()

@mcp.tool()
async def calculator(expression: str) -> str:
    """_summary_

    Args:
        expression (str): 符合 Python eval() 函式調用的數學表達式

    Returns:
        str: 表達式的計算結果
    """
    try:
        result = eval(expression)
        return json.dumps({"result": result})
    except Exception as e:
        return json.dumps({"result": f"數學表達式計算出錯: {str(e)}"})

if __name__ == "__main__":
    mcp.run(transport="stdio")
