from langchain_openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

llm = OpenAI(model = "gpt-3.5-turbo")

result = llm.invoke("What is the capital of France?")
print(result)

# i don't buy the openai package of 5 dollar so this is not working