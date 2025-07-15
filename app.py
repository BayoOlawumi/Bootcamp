# Import basic libraries
from dotenv import load_dotenv
import os
import streamlit as st

import sys
import subprocess

#st.write("Python executable:", sys.executable)
#st.write("Python version:", sys.version)
#st.write("Installed packages:")
#st.code(subprocess.check_output([sys.executable, "-m", "pip", "freeze"]).decode("utf-8"))


load_dotenv()

# import variables to this environment
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
os.environ["LANGCHAIN_TRACING_V2"] = "true"
project = os.getenv("LANGCHAIN_PROJECT")
if project is not None:
    os.environ["LANGCHAIN_PROJECT"] = project
else:
    print("LANGCHAIN_PROJECT environment variable is not set.")


# Import other libraries
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import OllamaLLM
from langchain_openai import ChatOpenAI
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
from langchain_community.document_loaders import UnstructuredPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.output_parsers import StrOutputParser





# Prompt Template
prompt = ChatPromptTemplate.from_messages(
    [
        ("system","You are a helpful assistant for getting questions on python programming language"),
        ("user", "Question:{question}")
    ]
)

## streamlit framework
st.title("Demo of Python Questionaire App")
input_text = st.text_input("What is scope of the question you need")


## Ollama LLama2 model
# Installed Models: llama3.2:1b; deepseek-r1:1.5b
llm=OllamaLLM(model="llama3.2:1b")
output_parser = StrOutputParser()
chain = prompt|llm|output_parser


if input_text:
    st.write(
        chain.invoke({"question":input_text})
    )