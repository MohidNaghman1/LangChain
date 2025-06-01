from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
import os


load_dotenv()

huggingface_token = os.getenv("HUGGINGFACEHUB_API_TOKEN")

llm = HuggingFaceEndpoint(
    repo_id="HuggingFaceH4/zephyr-7b-beta",
    task="text-generation",
    huggingfacehub_api_token=huggingface_token

)


model = ChatHuggingFace(llm=llm)

# 1st prompt -> detailed report
template1 = PromptTemplate(
    template='Write a detailed report on {topic}',
    input_variables=['topic']
)

# 2nd prompt -> summary
template2 = PromptTemplate(
    template='Write a 5 line summary on the following text. /n {text}',
    input_variables=['text']
)

parser = StrOutputParser()

# prompt1 = template1.invoke({'topic':'black hole'})

# result = model.invoke(prompt1)

# # Parse the output of the first model
# result = parser.parse(result.content)

# prompt2 = template2.invoke({'text':result})

# result1 = model.invoke(prompt2)
# # Parse the output of the second model
# result1 = parser.parse(result1.content)

# print(result1)

# do this work by the concept of chaining

chain = template1 | model | parser | template2 | model | parser
result = chain.invoke({'topic': 'black hole'})

print(result)