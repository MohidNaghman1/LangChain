from langchain_huggingface import ChatHuggingFace
from langchain_huggingface.llms import HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.schema.runnable import RunnableParallel
from dotenv import load_dotenv
import os


# 1) Load your HF API key (make sure HUGGINGFACEHUB_API_TOKEN is set in .env)
load_dotenv()
hf_token = os.getenv("HUGGINGFACEHUB_API_TOKEN")

llm_zephyr = HuggingFaceEndpoint(
    repo_id="HuggingFaceH4/zephyr-7b-beta",
    task="text-generation",
    huggingfacehub_api_token=hf_token
)
chat_model_zephyr = ChatHuggingFace(llm=llm_zephyr)


llm = HuggingFaceEndpoint(
    repo_id="HuggingFaceH4/zephyr-7b-beta",
    task="text-generation",
    huggingfacehub_api_token=hf_token
)


chat2= ChatHuggingFace(llm=llm)


prompt = PromptTemplate(
    template = "Generate short and simple notes from the following text: {text}",
    input_variables=["text"]
)

prompt2 = PromptTemplate(
    template = "Generate 5 short quetion answer from the following text: {text}",
    input_variables=["text"]    
)

prompt3 = PromptTemplate(
    template="Merge the provided notes and questions into a single coherent document: {notes} {quiz}",
    input_variables=["notes", "quiz"]
)

parser = StrOutputParser()

parallel_chain = RunnableParallel(
    {
        "notes": prompt | chat_model_zephyr | parser,
        "quiz": prompt2 | chat2 | parser
    }
)

merge_chain = prompt3 | chat_model_zephyr | parser

chain = parallel_chain | merge_chain


text = """
Artificial Intelligence (AI) is the simulation of human
 intelligence processes by machines, especially computer systems.
   These processes include learning 
(the acquisition of information and rules for using it),
 reasoning (using rules to reach approximate or definite conclusions), 
 and self-correction. AI has applications in various fields such as healthcare,
   finance, and transportation, where it can improve efficiency and decision-making.
"""
result = chain.invoke({
"text": text
})

print(result)