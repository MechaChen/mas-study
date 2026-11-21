import os
import uuid
import subprocess

from mcp.server import MCPServer

mcp = MCPServer(name="代碼解釋器")
BASE_DIR = "/Users/claire/Desktop/Benson/mas-study"
UV_CMD = "uv"

@mcp.tool()
async def run_code(language: str, code: str, timeout: int = 30) -> str:
    """根據語言運行代碼並返回執行結果，Python 代碼會執行 /Users/claire/Desktop/Benson/mas-study
中 uv 創建的 Python 3

    Args:
        language (str): 'python' 或者 'node
        code (str): 要執行的代碼文本
        timeout (int, optional): 最長運行描述（默認為 30s)

    Returns:
        str: 執行輸出 (stdout) 或錯誤訊息 (stderr/異常)
    """
    # 1. 檢查傳遞的編程語言是否符合結果
    language = (language or "").strip().lower()
    if language not in ("python", "node"):
        return f"不支持的語言：{language}"

    # 2. 計算獲取臨時代碼文件名
    suffix = ".py" if language == "python" else ".js"
    name = f"temp_{uuid.uuid4().hex}-{suffix}" # 為什麼要用 hex？
    tmp_path = os.path.join(BASE_DIR, name)

    # 3. 確保目錄存在
    os.makedirs(BASE_DIR, exist_ok=True)

    try:
        # 4. 寫入臨時文件
        with open(tmp_path, "w", encoding="utf-8") as f:
            f.write(code)

        # 5. 判斷不同的語言類型執行不同的操作
        cwd = BASE_DIR
        if language == "python":
            # 6. 使用 uv 來運行對應文件
            cmd = [UV_CMD, "--directory", BASE_DIR, "run", name]
        else:
            # 7. 使用 node 命令運行腳本
            cmd = ["node", tmp_path]

        # 8. 調用子線程運行對應命令
        process = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=cwd,
        )

        # 9. 獲取輸出與錯誤結果
        stdout = process.stdout.strip()
        stderr = process.stderr.strip()

        # 10. 判斷狀態碼
        if process.returncode == 0:
            return stdout
        else:
            return f"命令返回非零狀態 {process.returncode}, stderr:\n{stderr or stdout}" # 沒有錯誤就輸出正常內容
    # 為什麼要把錯誤寫這麼詳細？ 因為 agent 如果清楚知道錯誤，就有機會自己修正重新執行
    except subprocess.TimeoutExpired:
        return f"執行超時(>{timeout}s)"
    except FileNotFoundError as e:
        return f"命令未找到活路徑錯誤: {str(e)}"
    except Exception as e:
        return f"執行異常: {str(e)}"
    finally:
        # 嘗試刪除臨時文件（並且忽略）
        try:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
        except Exception:
            pass

if __name__ == "__main__":
    mcp.run(transport="streamable-http", port=9888)