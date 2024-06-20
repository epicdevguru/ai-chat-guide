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
from streamlit_webrtc import webrtc_streamer, WebRtcMode, AudioProcessorBase
import soundfile as sf
import numpy as np

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

@st.cache(allow_output_mutation=True)
def get_llm():
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

prompt: str = st.chat_input("Enter a prompt here")

client = OpenAI()

# AudioProcessor class for WebRTC
class AudioProcessor(AudioProcessorBase):
    def __init__(self):
        self.audio_frames = []

    def recv(self, frame):
        self.audio_frames.append(frame.to_ndarray().flatten())
        return frame

def recognize_speech_with_openai(audio_bytes):
    transcription = client.audio.transcriptions.create(
        model="whisper-1",
        file=audio_bytes,
        response_format="text"
    )
    return transcription['text']

webrtc_ctx = webrtc_streamer(
    key="audio-recorder",
    mode=WebRtcMode.SENDONLY,
    audio_processor_factory=AudioProcessor,
    async_processing=True,
)

if webrtc_ctx.state.playing:
    st.write("Recording...")

    if st.button("Stop Recording"):
        if webrtc_ctx.audio_processor:
            audio_frames = webrtc_ctx.audio_processor.audio_frames
            if audio_frames:
                audio_frames = np.concatenate(audio_frames)
                audio_bytes = io.BytesIO()
                sf.write(audio_bytes, audio_frames, 44100, format='wav')
                audio_bytes.seek(0)
                try:
                    prompt = recognize_speech_with_openai(audio_bytes)
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
                except Exception as e:
                    st.error(f"Error recognizing speech: {e}")
            else:
                st.write("No audio frames captured. Please try recording again.")
        else:
            st.write("Audio processor not initialized. Please try again.")

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
