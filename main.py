
#----------- for deployment
__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
#-----------

#------------ for local run
#from dotenv import load_dotenv
#load_dotenv()
#------------
from dataclasses import dataclass
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_openai import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.docstore.document import Document
#from langchain.chains import LLMChain

import json
import streamlit as st;


st.title("Chat AI agent")
st.write("---")

with open("gikAiJSONFile.json", "r", encoding='utf-8') as f:
    data = f.read()
    if data.startswith('\ufeff'):
        data = data[1:]
jsonData = json.loads(data)
embedding_function = OpenAIEmbeddings()
documents = [Document(page_content=item, metadata={"source": "gikAiJSONFile.json"}) for item in jsonData["dishes"]]
db = Chroma.from_documents(documents, embedding_function)

# loader = PyPDFLoader("vertopal.com_gikAiJSONFile.pdf")
# pages = loader.load_and_split()

# text_splitter = RecursiveCharacterTextSplitter(
#     # Set a really small chunk size, just to show.
#     chunk_size=300,
#     chunk_overlap=20,
#     length_function=len,
#     is_separator_regex=False,
# )

# texts = text_splitter.split_documents(pages)
# embeddings_model = OpenAIEmbeddings()
# db = Chroma.from_documents(texts, embeddings_model, persist_directory="./chroma.db")

@dataclass
class Message:
    actor: str
    payload: str

USER = "user"
ASSISTANT = "ai"
MESSAGES = "messages"

def get_llm() -> ChatOpenAI:
    return ChatOpenAI(model_name="gpt-4o",temperature=0)

def initialize_session_state():
    if MESSAGES not in st.session_state:
        st.session_state[MESSAGES] = [Message(actor=ASSISTANT, payload="Hai! Saya Ika, selamat datang di Galeri Indonesia Kaya. Butuh info tentang Indonesia? Tanya saja! Saya siap membantu.")]
    if "llm_chain" not in st.session_state:
        st.session_state["llm_chain"] = RetrievalQA.from_chain_type(get_llm(), retriever=db.as_retriever())#get_llm_chain()

def get_llm_chain_from_session():
    return st.session_state["llm_chain"]

initialize_session_state()

msg: Message
for msg in st.session_state[MESSAGES]:
    st.chat_message(msg.actor).write(msg.payload)

prompt: str = st.chat_input("Enter a prompt here")

if prompt:
    st.session_state[MESSAGES].append(Message(actor=USER, payload=prompt))
    st.chat_message(USER).write(prompt)

    with st.spinner("Please wait..."):
        llm_chain = get_llm_chain_from_session()
        response: str = llm_chain({"query": prompt})["result"]
        st.session_state[MESSAGES].append(Message(actor=ASSISTANT, payload=response))
        st.chat_message(ASSISTANT).write(response)


# question = st.text_input('Ask a question.')#"What is the content of this document?"
# if st.button('Submit'):
#     with st.spinner('Wait for it ...'):
#         llm = ChatOpenAI(model_name="gpt-3.5-turbo",temperature=0)
#         # retriever_from_llm = MultiQueryRetriever.from_llm(
#         #    retriever=db.as_retriever(), llm=llm
#         # )
#         #docs = retriever_from_llm.get_relevant_documents(query=question)
#         qa_chain = RetrievalQA.from_chain_type(llm, retriever=db.as_retriever())
#         result = qa_chain({"query": question})
#         st.write(result["result"])


