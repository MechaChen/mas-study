import dotenv
from openai import OpenAI
from pydantic import BaseModel, Field

dotenv.load_dotenv()

class SplitTask(BaseModel):
	task_count: int = Field(..., gt=0, le=10, description="任務數量")
	tasks: list[str] = Field(..., description="任務拆分列表")
	
client = OpenAI()

system_prompt = """用戶將提一個問題，請將任務拆成多個小任務，數量介於 1-10 個，且必須以 json 的格式輸出，其中包含 tasks：一個陣列，包含所以用字串表示拆分的小任務，task_count：全部小任務的總量
"""

while True:
	query = input('\nQuery: ')
	
	if query.lower() == 'quit':
		break

	response = client.chat.completions.create(
		model="deepseek-chat",
		messages=[
			{"role": "system", "content": system_prompt},
			{"role": "user", "content": query}
		],
		response_format={
			"type": "json_object"
		}
	)
	
	response_message = response.choices[0].message
	split_tasks = SplitTask.model_validate_json(response_message.content)

	print(f"\n任務數量： {split_tasks.task_count}\n")

	for idx, task in enumerate(split_tasks.tasks):
		print(f"{str(idx + 1).zfill(2)}. {task}")
		
	print('\n================')