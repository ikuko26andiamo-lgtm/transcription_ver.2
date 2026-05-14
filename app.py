import streamlit as st
import numpy as np
import queue
import docx
import google.generativeai as genai
from streamlit_webrtc import webrtc_streamer, WebRtcMode, AudioProcessorBase
from faster_whisper import WhisperModel
import torch
import threading

# --- モデルロード ---
@st.cache_resource
def load_whisper_model():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    return WhisperModel("base", device=device, compute_type="float16" if device=="cuda" else "int8")

# --- 音声処理クラス（キュー方式に全面改修） ---
class RealTimeGeminiProcessor(AudioProcessorBase):
    def __init__(self, whisper_model, api_key, model_name, terms, persona):
        self.whisper_model = whisper_model
        self.terms = terms
        self.persona = persona
        self.audio_queue = queue.Queue() # 音声データを一時保管する場所
        self.result_queue = queue.Queue() # Geminiの結果を入れる場所
        genai.configure(api_key=api_key)
        self.gemini_model = genai.GenerativeModel(model_name)
        
        # 裏側で文字起こしを回し続けるスレッドを開始
        self.processing_thread = threading.Thread(target=self._process_loop, daemon=True)
        self.processing_thread.start()

    def recv(self, frame):
        # ここでは音声を受け取ってキューに入れる「だけ」にする（超高速）
        audio = frame.to_ndarray()
        if audio.ndim > 1: audio = np.mean(audio, axis=1)
        
        # 48kHz -> 16kHz
        sample_rate = frame.sample_rate
        if sample_rate != 16000:
            step = sample_rate // 16000
            audio = audio[::step]

        self.audio_queue.put(audio.flatten().astype(np.float32) / 32768.0)
        return frame

    def _process_loop(self):
        # 裏側（別スレッド）でじっくりAI処理を行う
        buffer = []
        while True:
            # キューから音声のかけらを取り出す
            audio_chunk = self.audio_queue.get()
            buffer.extend(audio_chunk)

            # 5秒分溜まったらAIを叩く
            if len(buffer) >= 16000 * 5:
                segment_audio = np.array(buffer)
                buffer = [] # バッファを空にする
                
                try:
                    segments, _ = self.whisper_model.transcribe(segment_audio, language="ja")
                    raw_text = "".join([s.text for s in segments]).strip()
                    
                    if raw_text:
                        p = f"修正して: {raw_text}\n用語: {','.join(self.terms[:20])}"
                        response = self.gemini_model.generate_content(p)
                        self.result_queue.put(response.text.strip())
                except:
                    pass

# --- UI部分 ---
st.title("🎙️ リアルタイム講義補正ノート")
# ... (サイドバーやsession_stateの初期化は以前のコードと同じ) ...

# 🔴 以下、WebRTC設定の修正
model_name_val = st.session_state.model_name
terms_val = st.session_state.terms

def audio_processor_factory():
    return RealTimeGeminiProcessor(
        whisper_model=load_whisper_model(),
        api_key=api_key,
        model_name=model_name_val,
        terms=terms_val,
        persona=persona
    )

webrtc_ctx = webrtc_streamer(
    key="lecture-gemini",
    mode=WebRtcMode.SENDONLY,
    media_stream_constraints={"video": False, "audio": True},
    rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
    audio_processor_factory=audio_processor_factory,
    # 🔴 async_processingをTrueに設定して、メインスレッドの詰まりを防ぐ
    async_processing=True, 
)

# 画面表示ループはそのまま
