from langchain_huggingface import ChatHuggingFace
from langchain_huggingface.llms import HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.schema.runnable import RunnableParallel,RunnableBranch,RunnableLambda
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel,Field
from typing import Literal
from dotenv import load_dotenv
import os


# 1) Load your HF API key (make sure HUGGINGFACEHUB_API_TOKEN is set in .env)
load_dotenv()
hf_token = os.getenv("HUGGINGFACEHUB_API_TOKEN")

llm = HuggingFaceEndpoint(
    repo_id="HuggingFaceH4/zephyr-7b-beta",
    task="text-generation",
    huggingfacehub_api_token=hf_token
)
chat_model_zephyr = ChatHuggingFace(llm=llm)

parser = StrOutputParser()

class Feedback(BaseModel):
    sentiment: Literal["positive", "negative"]= Field(description="give the sentiment of the text")

parser2 = PydanticOutputParser(pydantic_object=Feedback)

prompt = PromptTemplate(
    template="classify the sentiment of the following text into positive or negative: {text} \n {format_instructions}",
    input_variables=["text"],
    partial_variables={"format_instructions": parser2.get_format_instructions()}
)

classifer_chain = prompt | chat_model_zephyr | parser2

prompt2 = PromptTemplate(
    template="Write an appropiate response to this positive feedback: {feedback}",
    input_variables=["feedback"]
)

prompt3= PromptTemplate(
    template="Write an appropiate response to this negative feedback: {feedback}",
    input_variables=["feedback"]
)

branch_chain = RunnableBranch(
(lambda x: x.sentiment == "positive", prompt2| chat_model_zephyr | parser),
(lambda x: x.sentiment == "negative", prompt3 | chat_model_zephyr | parser),
RunnableLambda(lambda x: "I'm sorry, I don't understand")
)

chain = classifer_chain | branch_chain

text = "I love this product"
result = chain.invoke({"text": text})

print(result)