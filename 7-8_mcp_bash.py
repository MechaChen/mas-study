import subprocess

from mcp.server import MCPServer

mcp = MCPServer(name="Bash 工具")

@mcp.tool()
async def bash(command: str) -> dict:
    """傳遞 command 命令，在

    Args:
        command (str): 需要執行的 command 命令

    Returns:
        dict: 返回命令的執行狀態、結果、錯誤訊息
    """
    result = subprocess.run(
        command,
        shell=True, # 讓命令通過 cmd 執行
        capture_output=True, # 捕獲輸出
        text=True, # 輸出解碼為字符串
    )

    return {
        "returncode": result.returncode,
        "stdout": result.stdout,
        "stderr": result.stderr
    }

if __name__ == "__main__":
    mcp.run(transport="stdio")