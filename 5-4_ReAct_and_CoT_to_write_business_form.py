import json
from datetime import date
from typing import Literal

import dotenv
from openai import OpenAI
from openai.types.chat.chat_completion_chunk import ChoiceDeltaToolCall
from pydantic import BaseModel, Field

dotenv.load_dotenv()

class GetEmployeeInfoInput(BaseModel):
  employee_id: str = Field(..., description="員工編號")

class SubmitReimbursementInput(BaseModel):
  employee_id: str = Field(..., description="員工編號")
  employee_name: str = Field(..., description="員工姓名")
  submission_date: date = Field(..., description="表單提交日期")
  trip_start_date: date = Field(..., description="出差起始日期")
  trip_end_date: date = Field(..., description="出差結束日期")
  destination: str = Field(..., description="出差目的地")
  transportation_cost: float = Field(..., description="交通費用")
  accommodation_cost: float = Field(..., description="住宿費用")
  meal_cost: float = Field(..., description="餐費")
  reimbursement_level: Literal["標準", "高級", "VIP"] = Field(..., description="報銷等級")

class CalculatorInput(BaseModel):
  expression: str = Field(..., description="程式可執行的數學表達式")


def get_employee_info(employee_id: str) -> str:
  """根據員工編號，查詢員工的訊息，像是名稱、職級"""
  if employee_id == 'E12345':
    return json.dumps({"name": "敬岳", "level": "資深工程師"})

  return json.dumps({"error": "員工編號不存在"})


def submit_reimbursement(**reimbursement_inputs) -> str:
  """提交已填寫完整的出差報銷表單"""
  print('-----已提交報銷表單-----')
  print(reimbursement_inputs)

  return json.dumps({"success": True, "messages": "報銷表單已提交成功"})


def calculator(expression: str) -> str:
  """一個計算機，可以執行程式可執行的數學表達式"""
  try:
    result = eval(expression)
    return json.dumps({"result": result})
  except Exception as e:
    return f"Expression Error: {str(e)}"

SYSTEM_PROMPT = """
你是一個智能企業報銷單助手。你的任務是根據用戶的請求和公司政策，幫助員工填寫並提交差旅報銷單

你必須遵循以下思考和行動模式 (ReAct):

***思考(Thought)***:
  - **回顧目標**: 當前我的最終目標是什麼？（例如：請填寫並提交一份完整的報銷單)
  - **分析現狀**: 我已經獲取了哪些訊息？ 還缺少哪些訊息？
  - **運用 CoT(Chain of Thought)**: 仔細閱讀並一步步的應用報銷策略。例如，計算總金額、判斷報銷等級等。把你的計算和推理過程寫在思考中
  - **規劃下一步**: 接下來我該做什麼？ 是查詢訊息，還是準備提交報銷單？

***行動(Action)***:
	- 根據你的思考，決定調用工具還是向用戶提問
	- 如果所有的資訊都已經收集完畢，你的最後一部必須是調用 `submit_reinbursement` 工具
	
請開始工作
"""


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
    self.tools_map = {
      "get_employee_info": {
        "tool": get_employee_info,
        "input": GetEmployeeInfoInput,
      },
      "submit_reimbursement": {
        "tool": submit_reimbursement,
        "input": SubmitReimbursementInput,
      },
      "calculator": {
        "tool": calculator,
        "input": CalculatorInput,
      }
    }
    self.available_tools = [
      {
        "type": "function",
        "function": {
          "name": tool["tool"].__name__,
          "description": tool["tool"].__doc__,
          "parameters": tool["input"].model_json_schema()
        }
      } for tool in self.tools_map.values()
    ]

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
        tool_result = self.tools_map[tool_name]["tool"](**tool_arguments)

        print(f'\n\nTool name: {tool_name}')
        print(f'Tool result: {tool_result}')

        self.messages.append({
          "tool_call_id": tool_call.id,
          "role": "tool",
          "name": tool_name,
          "content": tool_result
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
