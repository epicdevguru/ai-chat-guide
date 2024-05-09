
#----------- for deployment
__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
#-----------

#------------ for local run
#from dotenv import load_dotenv
#load_dotenv()
#------------
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_openai import ChatOpenAI
from langchain.chains import RetrievalQA
import streamlit as st;


st.title("Chat AI agent")
st.write("---")

loader = PyPDFLoader("Pariwisata.pdf")
pages = loader.load_and_split()

text_splitter = RecursiveCharacterTextSplitter(
    # Set a really small chunk size, just to show.
    chunk_size=300,
    chunk_overlap=20,
    length_function=len,
    is_separator_regex=False,
)

texts = text_splitter.split_documents(pages)
print(texts[0])
embeddings_model = OpenAIEmbeddings()
db = Chroma.from_documents(texts, embeddings_model, persist_directory="./chroma.db")


question = st.text_input('Ask a question.')#"What is the content of this document?"
if st.button('Submit'):
    with st.spinner('Wait for it ...'):
        llm = ChatOpenAI(model_name="gpt-3.5-turbo",temperature=0)
        # retriever_from_llm = MultiQueryRetriever.from_llm(
        #    retriever=db.as_retriever(), llm=llm
        # )
        #docs = retriever_from_llm.get_relevant_documents(query=question)
        qa_chain = RetrievalQA.from_chain_type(llm, retriever=db.as_retriever())
        result = qa_chain({"query": question})
        st.write(result["result"])
