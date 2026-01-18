import time

def make_coffee(customer_name: str) -> None:
  print(f'開始為 {customer_name} 製作咖啡...')
  time.sleep(5)
  print(f'{customer_name} 的咖啡煮好了')

def main_sync() -> None:
  start_time = time.time()

  make_coffee('顧客 A')
  make_coffee('顧客 B')
  make_coffee('顧客 C')
  make_coffee('顧客 D')
  make_coffee('顧客 E')
  make_coffee('顧客 F')

  end_time = time.time()

  print(f'同步方法總共煮咖啡時間: {end_time - start_time:.2f} 秒')

if __name__ == '__main__':
  main_sync()