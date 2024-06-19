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

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
from pathlib import Path
from openai import OpenAI
import json
import streamlit as st
import io
from pydub import AudioSegment
import ffmpeg
from tempfile import NamedTemporaryFile

st.title("Chat AI agent")
st.write("---")

with open("gikAiJSONFile.json", "r", encoding='utf-8') as f:
    data = f.read()
    if data.startswith('\ufeff'):
        data = data[1:]
jsonData = json.loads(data)
embedding_function = OpenAIEmbeddings()
documents = [Document(page_content=str(item), metadata={"source": "gikAiJSONFile.json"}) for item in jsonData["dishes"]]
db = Chroma.from_documents(documents, embedding_function)

@dataclass
class Message:
    actor: str
    payload: str

USER = "user"
ASSISTANT = "ai"
MESSAGES = "messages"

def get_llm() -> ChatOpenAI:
    return ChatOpenAI(model_name="gpt-4", temperature=0)

def initialize_session_state():
    if MESSAGES not in st.session_state:
        st.session_state[MESSAGES] = [Message(actor=ASSISTANT, payload="Hai! Saya Ika, selamat datang di Galeri Indonesia Kaya. Butuh info tentang Indonesia? Tanya saja! Saya siap membantu.")]
    if "llm_chain" not in st.session_state:
        st.session_state["llm_chain"] = RetrievalQA.from_chain_type(get_llm(), retriever=db.as_retriever())

def get_llm_chain_from_session():
    return st.session_state["llm_chain"]

initialize_session_state()

msg: Message
for msg in st.session_state[MESSAGES]:
    st.chat_message(msg.actor).write(msg.payload)

client = OpenAI()

# Function to record audio using pydub and save it to a file
def record_audio(file_path, duration=5, fs=44100):
    import pyaudio
    import wave

    p = pyaudio.PyAudio()
    stream = p.open(format=pyaudio.paInt16, channels=1, rate=fs, input=True, frames_per_buffer=1024)
    frames = []

    st.write("Recording...")

    for _ in range(0, int(fs / 1024 * duration)):
        data = stream.read(1024)
        frames.append(data)

    st.write("Recording completed.")

    stream.stop_stream()
    stream.close()
    p.terminate()

    wf = wave.open(file_path, 'wb')
    wf.setnchannels(1)
    wf.setsampwidth(p.get_sample_size(pyaudio.paInt16))
    wf.setframerate(fs)
    wf.writeframes(b''.join(frames))
    wf.close()

# Function to recognize speech using the OpenAI API
def recognize_speech_with_openai(audio_path):
    audio_file = open(audio_path, "rb")
    transcription = client.audio.transcriptions.create(
        model="whisper-1",
        file=audio_file,
        response_format="text"
    )
    return transcription['text']

prompt: str = st.chat_input("Enter a prompt here or use the microphone")

# Add a button to capture speech input
if st.button("Record from Microphone"):
    audio_file_path = "speech.wav"
    record_audio(audio_file_path)
    speech_text = recognize_speech_with_openai(audio_file_path)
    if speech_text:
        prompt = speech_text
        st.write(f"Recognized text: {prompt}")

if prompt:
    st.session_state[MESSAGES].append(Message(actor=USER, payload=prompt))
    st.chat_message(USER).write(prompt)

    with st.spinner("Please wait..."):
        llm_chain = get_llm_chain_from_session()
        dataResponse: str = llm_chain({"query": prompt})["result"]
        response = client.audio.speech.create(
            model="tts-1",
            voice="alloy",
            input=dataResponse,
        )
        
        st.session_state[MESSAGES].append(Message(actor=ASSISTANT, payload=dataResponse))
        st.chat_message(ASSISTANT).write(dataResponse)

        # Play the audio directly from the binary content
        audio_bytes = io.BytesIO(response.content)
        st.audio(audio_bytes, format="audio/mp3")
