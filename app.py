import streamlit as st
from openai import OpenAI
from streamlit_webrtc import webrtc_streamer, WebRtcMode, AudioProcessorBase
import io

st.title("Chat AI Agent with Text and Audio Input")

client = OpenAI()

@st.cache(allow_output_mutation=True)
def get_llm():
    return ChatOpenAI(model_name="gpt-4", temperature=0)

def initialize_session_state():
    if "messages" not in st.session_state:
        st.session_state["messages"] = [
            {"actor": "ai", "payload": "Hi! I am here to assist you with information about Indonesia. How can I help you?"}
        ]
    if "llm_chain" not in st.session_state:
        st.session_state["llm_chain"] = get_llm()

class AudioProcessor(AudioProcessorBase):
    def __init__(self):
        self.audio_frames = []

    def recv(self, frame):
        self.audio_frames.append(frame.to_ndarray().flatten().astype("float32"))
        return frame

def main():
    initialize_session_state()

    st.header("Chat with AI")

    # Input for text prompt
    prompt = st.text_input("Enter a prompt here")

    webrtc_ctx = webrtc_streamer(
        key="audio-recorder",
        mode=WebRtcMode.SENDONLY,
        audio_processor_factory=AudioProcessor,
        async_processing=True,
    )

    if webrtc_ctx.state.playing:
        st.write("Recording...")

        # Button to stop recording
        if st.button("Stop Recording"):
            if webrtc_ctx.audio_processor:
                audio_frames = webrtc_ctx.audio_processor.audio_frames
                if audio_frames:
                    audio_bytes = io.BytesIO()
                    wavio.write(audio_bytes, audio_frames, 44100, sampwidth=2)
                    audio_bytes.seek(0)
                    prompt = recognize_speech_with_openai(audio_bytes)
                    st.write(f"Recognized text: {prompt}")

                    if prompt:
                        process_prompt(prompt)

    if prompt:
        process_prompt(prompt)

def process_prompt(prompt):
    st.session_state["messages"].append({"actor": "user", "payload": prompt})
    st.chat_message("user").write(prompt)

    with st.spinner("Please wait..."):
        llm_chain = st.session_state["llm_chain"]
        response = llm_chain({"query": prompt})["result"]
        st.session_state["messages"].append({"actor": "ai", "payload": response})
        st.chat_message("ai").write(response)

        # Play the audio directly from the binary content
        audio_bytes = io.BytesIO(response.content)
        st.audio(audio_bytes, format="audio/mp3")

def recognize_speech_with_openai(audio_bytes):
    transcription = client.audio.transcriptions.create(
        model="whisper-1",
        file=audio_bytes,
        response_format="text"
    )
    return transcription['text']

if __name__ == "__main__":
    main()
