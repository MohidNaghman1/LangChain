from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv

load_dotenv()

embeddings = OpenAIEmbeddings(model="text-embedding-3-small",dimension=32)

documents = [
    "This is a sample document.",
    "This is another sample document.",
    "This is a third sample document."
]

result = embeddings.embed_documents(documents)

print(result)