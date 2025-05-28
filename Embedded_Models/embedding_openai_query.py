from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv

load_dotenv()

embeddings = OpenAIEmbeddings(model="text-embedding-3-small",dimension=32)

result = embeddings.embed_query("Islamabad is the capital of Pakistan")

print(result)