from langchain_huggingface import ChatHuggingFace
from langchain_huggingface.llms import HuggingFaceEndpoint
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
import os

# 1) Load your HF API key (make sure HUGGINGFACEHUB_API_TOKEN is set in .env)
load_dotenv()
hf_token = os.getenv("HUGGINGFACEHUB_API_TOKEN")

# 2) Wrap Zephyr with HuggingFaceEndpoint
llm = HuggingFaceEndpoint(
    repo_id="HuggingFaceH4/zephyr-7b-beta",
    task="text-generation",
    huggingfacehub_api_token=hf_token,
)

# 3) Create a ChatHuggingFace instance
chat_model = ChatHuggingFace(llm=llm)

# 4) Build a ChatPromptTemplate
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant."),
    ("human", "{input}"),
])

# 5) Attach an output parser (just returns raw string here)
parser = StrOutputParser()

# 6) Chain them
chain = prompt | chat_model | parser

# 7) Helper function
def zephyr_chat(query: str) -> str:
    return chain.invoke({"input": query})


print(zephyr_chat("Tell me a fun fact about space."))
