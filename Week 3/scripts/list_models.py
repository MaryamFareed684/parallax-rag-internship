import os
import sys

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
client = OpenAI(api_key=os.getenv("LLM_API_KEY"), base_url=os.getenv("LLM_BASE_URL"))

keyword = sys.argv[1].lower() if len(sys.argv) > 1 else ""
for m in sorted(client.models.list().data, key=lambda m: m.id):
    if keyword in m.id.lower():
        print(m.id)