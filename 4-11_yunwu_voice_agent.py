import os
import numpy
import soundfile
import sounddevice
from pynput import keyboard as pynput_keyboard
import tempfile
import json

import dotenv

from openai import OpenAI

dotenv.load_dotenv()

yumwu_base_url = os.getenv('YUMWU_BASE_URL')
yumwu_api_key = os.getenv('YUMWU_API_KEY')

def calculator(expression: str) -> dict:
  try:
    result = eval(expression)
    return {"result": result}
  except Exception as e:
    return {"error": f'不支援的表達式: {str(e)}'}

class ReActAgent:
  def __init__(self):
    self.model="deepseek-chat"
    self.client = OpenAI()
    self.messages = [
      {
        "role": "system",
        "content": "你是一個智慧助手，你會優先使用工具來幫你解決問題 & 回答問題，另外如果不知道就說無法回答"
      }
    ]
    self.available_tools = [
      {
        "type": "function",
        "function": {
          "name": "calculator",
          "description": "一個可以執行數學表達式並回傳結果的工具",
          "parameters": {
            "type": "object",
            "properties": {
              "expression": {
                "type": "string",
                "description": "數學表達式，例如 123 + 456 * 789"
              }
            }
          }
        }
      }
    ]
    self.tools = {"calculator": calculator}
    

  def speech_to_text(self) -> str:
    # 1. initialize recording, is_recording, samplerate, channels
    # 2. get recording from sounddevice
    # 3. convert recording to numpy array
    # 4. save recording numpy array to wav file by soundfile and tempfile
    # 5. get transcription from YumWu whisper model by passing wav file path
    recording = []
    is_recording = False
    samplerate = 16000
    channels = 1

    def callback(indata, frames, time, status):
      nonlocal is_recording, recording
      if is_recording:
        recording.append(indata.copy())

    stream = sounddevice.InputStream(
      samplerate=samplerate,
      channels=channels,
      callback=callback
    )
    stream.start()
    print('按下空白鍵開始錄音')

    def on_press(key):
      if key == pynput_keyboard.Key.space:
        return False

    with pynput_keyboard.Listener(on_press=on_press) as listener:
      listener.join()

    is_recording = True
    print('開始錄音，再次按下空白鍵結束錄音')

    with pynput_keyboard.Listener(on_press=on_press) as listener:
      listener.join()
    stream.stop()
    stream.close()
    is_recording = False

    if not recording:
      print('沒有錄到任何聲音')
      return ""

    audio_data = numpy.concatenate(recording, axis=0)

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
      soundfile.write(
        temp_file.name,
        audio_data,
        samplerate,
      )
      audio_path = temp_file.name

    with open(audio_path, "rb") as audio_file:
      audio_client = OpenAI(
        base_url=yumwu_base_url,
        api_key=yumwu_api_key,
      )
      transcripts = audio_client.audio.transcriptions.create(
        model="whisper-1",
        file=audio_file,
      )

    return transcripts.text

  def text_to_speech(self, text: str) -> None:
    # 1. convert text to audio data by tts model
    # 2. save audio data to wav file by soundfile and tempfile
    # 3. play audio data by sounddevice
    audio_client = OpenAI(
      base_url=yumwu_base_url,
      api_key=yumwu_api_key
    )

    response = audio_client.audio.speech.create(
      voice="alloy",
      model="tts-1",
      input=text,
    )

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
      temp_file.write(response.read())
      audio_path = temp_file.name

    data, samplerate = soundfile.read(audio_path)
    sounddevice.play(data, samplerate)
    sounddevice.wait()

  def process_query(self, query: str) -> str:
    # 1. add user query to message history
    # 2. ask DeepSeek with tools selection
    # 3. add 1st response to message history
    # 4. if having tool calls, loop the tool call, get the tool result and add to message history
    # 5. ask DeepSeek 2nd time to get summary without tools selection
    self.messages.append({
      "role": "user",
      "content": query,
    })

    response = self.client.chat.completions.create(
      model=self.model,
      messages=self.messages,
      tools=self.available_tools,
      tool_choice="auto",
    )

    response_message = response.choices[0].message

    tool_calls = response_message.tool_calls
    self.messages.append(response_message.model_dump())

    if tool_calls:
      for tool_call in tool_calls:
        tool_name = tool_call.function.name
        tool_args = json.loads(tool_call.function.arguments)
        function_to_call = self.tools[tool_name]

        tool_result = function_to_call(**tool_args)
        self.messages.append({
          "tool_call_id": tool_call.id,
          "role": "tool",
          "name": tool_name,
          "content": json.dumps(tool_result)
        })

      second_response = self.client.chat.completions.create(
        model=self.model,
        messages=self.messages,
        tools=self.available_tools,
        tool_choice="none",
      )

      second_response_message = second_response.choices[0].message
      self.messages.append(second_response_message.model_dump())

      return second_response_message.content
    else:
      return response_message.content

  def chat_loop(self) -> None:
    while True:
      try:
        query = self.speech_to_text().strip()
        print(f"Query: {query}")
        
        if query.lower() == '退出':
          break  

        answer = self.process_query(query)
        print(f"Assistant: {answer}")
        self.text_to_speech(answer)
        
      except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
  agent = ReActAgent()
  agent.chat_loop()