import os
import time

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
client = OpenAI(api_key=os.getenv("LLM_API_KEY"), base_url=os.getenv("LLM_BASE_URL"), timeout=30)

start = time.perf_counter()
response = client.chat.completions.create(
    model=os.getenv("LLM_MODEL"),
    messages=[{"role": "user", "content": "Say hello in one short sentence."}],
    max_tokens=300,
)
choice = response.choices[0]
print("Answer:", choice.message.content)
print("Finish reason:", choice.finish_reason)
print("Token usage:", response.usage)
print(f"Took {(time.perf_counter() - start) * 1000:.0f} ms")