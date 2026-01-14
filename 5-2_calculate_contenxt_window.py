import transformers
import jinja2

tokenizer = transformers.AutoTokenizer.from_pretrained(
  './deepseek_v3_tokenizer',
)

prompt = '你好，你是？'
messages = [
  {
    'role': 'user',
    'content': '幫我計算下 234 * 1123',
  }
]

print('prompt: ', len(tokenizer.encode(prompt)))
print('messages: ', len(tokenizer.apply_chat_template(messages))) # 這裡會用到 jinja2 的 template 語法
print('tokenized messages: ', tokenizer.apply_chat_template(messages)) # 這裡會用到 jinja2 的 template 語法

template = jinja2.Template(tokenizer.chat_template)

encoded_messages = template.render(messages=messages)
print('encoded messages: ', encoded_messages)