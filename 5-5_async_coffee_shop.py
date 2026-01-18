import asyncio
import time

async def make_coffee_async(customer_name: str) -> str:
  print(f'開始為 {customer_name} 煮咖啡...')
  await asyncio.sleep(5)
  print(f'{customer_name} 的咖啡煮好了')
  return f'{customer_name}的咖啡'

async def main_async() -> None:
  start_time = time.time()

  tasks = [
    make_coffee_async('顧客A'),
    make_coffee_async('顧客B'),
    make_coffee_async('顧客C'),
    make_coffee_async('顧客D'),
    make_coffee_async('顧客E'),
    make_coffee_async('顧客F'),
  ]

  results = await asyncio.gather(*tasks)
  print(f'所有咖啡都準備好了:', results)

  end_time = time.time()
  print(f'非同步方法煮咖啡總共時間：{end_time - start_time:.2f} 秒')

if __name__ == '__main__':
  asyncio.run(main_async())