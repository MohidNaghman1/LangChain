from langchain_huggingface import ChatHuggingFace
from langchain_huggingface.llms import HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
import os

# 1) Load your HF API key (make sure HUGGINGFACEHUB_API_TOKEN is set in .env)
load_dotenv()
hf_token = os.getenv("HUGGINGFACEHUB_API_TOKEN")

prompt1 = PromptTemplate(
    template="Generate a detailed report on the topic: {topic}",
    input_variables=["topic"]
)

prompt2 = PromptTemplate(
    template="Generate a 5 line summary of the following text: {text}",
    input_variables=["text"]
)

llm = HuggingFaceEndpoint(
    repo_id="HuggingFaceH4/zephyr-7b-beta",
    task="text-generation",
    huggingfacehub_api_token=hf_token
)

chat_model = ChatHuggingFace(llm=llm)

parser = StrOutputParser()

chain = prompt1 | chat_model | parser | prompt2 | chat_model | parser

result = chain.invoke(('topic', 'Artificial Intelligence and its impact on society.'))
print(result)