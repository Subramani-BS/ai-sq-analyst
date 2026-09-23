import os
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

client = Groq(api_key=os.getenv("GROQ_API_KEY"))
models = client.models.list()

print("\n✅ Available models on your account:\n")
for model in models.data:
    print(f"  {model.id}")