from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
import os


load_dotenv()

huggingface_token = os.getenv("HUGGINGFACEHUB_API_TOKEN")


model = HuggingFaceEndpoint(
    repo_id="google/flan-t5-small",  # lightweight free model
    task="text2text-generation",
    huggingfacehub_api_token=huggingface_token
)


model = ChatHuggingFace(llm=model)

parser = JsonOutputParser()

template = PromptTemplate(
    template='Give me 5 facts about {topic} \n {format_instruction}',
    input_variables=['topic'],
    partial_variables={'format_instruction': parser.get_format_instructions()}
)

# do this work by the concept of chaining
chain = template | model | parser
result = chain.invoke({'topic': 'black hole'})
print(result)