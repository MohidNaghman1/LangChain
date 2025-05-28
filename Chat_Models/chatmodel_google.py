from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv

load_dotenv()

model = ChatGoogleGenerativeAI(model="chat-bison-001")

result = model.invoke("What is the capital of France?")

print(result)