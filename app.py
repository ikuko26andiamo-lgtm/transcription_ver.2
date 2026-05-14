import streamlit as st
import numpy as np
import queue
import docx
import google.generativeai as genai
from streamlit_webrtc import webrtc_streamer, WebRtcMode, AudioProcessorBase
from faster_whisper import WhisperModel
import torch

# --- 設定とモデルロード ---
st.set_page_config(page_title="リアルタイム講義補正ノート📝", layout="wide")

@st.cache_resource
def load_whisper_model():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    return WhisperModel("base", device=device, compute_type="float16" if device=="cuda" else "int8")

def get_working_model(api_key):
    """利用可能なGeminiモデルを自動探索する"""
    try:
        genai.configure(api_key=api_key)
        available_models = [m.name for m in genai.list_models() if 'generateContent' in m.supported_generation_methods]
        
        candidates = ["models/gemini-1.5-flash", "models/gemini-1.5-pro", "models/gemini-pro"]
        for cand in candidates:
            if cand in available_models:
                return cand
        return available_models[0] if available_models else "models/gemini-1.5-flash"
    except:
        return "models/gemini-1.5-flash"

def extract_terms_with_gemini(file, api_key, model_name):
    doc = docx.Document(file)
    text = "\n".join([p.text for p in doc.paragraphs])
    context_text = text[:8000]
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel(model_name)
    prompt = f"以下の資料から重要な専門用語・固有名詞を最大50個抜き出してください。カンマ区切りで単語のみ出力してください。\n\n{context_text}"
    try:
        response = model.generate_content(prompt)
        return [t.strip() for t in response.text.split(",") if t.strip()]
    except:
        return ["法律", "政治", "憲法", "国際政治"]

# --- 音声 & Gemini 処理クラス ---
class RealTimeGeminiProcessor(AudioProcessorBase):
    # kwargsで渡す引数とここの引数名を完全に一致させる必要があります
    def __init__(self, whisper_model, api_key, model_name, terms, persona):
        self.whisper_model = whisper_model
        self.terms = terms
        self.persona = persona
        self.audio_buffer = []
        self.result_queue = queue.Queue()
        # Geminiの初期化
        genai.configure(api_key=api_key)
        self.gemini_model = genai.GenerativeModel(model_name)

    def recv(self, frame):
        audio = frame.to_ndarray().flatten().astype(np.float32) / 32768.0
        self.audio_buffer.extend(audio)

        if len(self.audio_buffer) >= 16000 * 3:
            segment_audio = np.array(self.audio_buffer)
            self.audio_buffer = []
            
            try:
                segments, _ = self.whisper_model.transcribe(
                    segment_audio, language="ja",
                    initial_prompt=f"専門用語: {','.join(self.terms[:10])}"
                )
                raw_text = "".join([s.text for s in segments]).strip()

                if raw_text:
                    # 補正
                    prompt = f"修正して: {raw_text}\nヒント: {','.join(self.terms[:30])}"
                    response = self.gemini_model.generate_content(prompt)
                    self.result_queue.put(response.text.strip())
            except:
                pass
        return frame

# --- UI ---
st.title("🎙️ リアルタイム講義補正ノート")

with st.sidebar:
    st.header("⚙️ 設定")
    api_key = st.text_input("Gemini API Key", type="password")
    persona = st.text_input("AIの役割", "法学部の教授")
    uploaded_docx = st.file_uploader("講義資料 (docx)", type="docx")

if not api_key or not uploaded_docx:
    st.info("APIキーと資料をセットしてください。")
    st.stop()

# モデルと用語の準備
if "model_name" not in st.session_state:
    st.session_state.model_name = get_working_model(api_key)
if "terms" not in st.session_state:
    st.session_state.terms = extract_terms_with_gemini(uploaded_docx, api_key, st.session_state.model_name)
if "full_notes" not in st.session_state:
    st.session_state.full_notes = ""

st.write(f"使用モデル: `{st.session_state.model_name}`")

# --- WebRTC 設定 (TypeErrorの主原因) ---
webrtc_ctx = webrtc_streamer(
    key="lecture-gemini",
    mode=WebRtcMode.SENDONLY,
    media_stream_constraints={"video": False, "audio": True},
    rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
    worker_class=RealTimeGeminiProcessor,
    # 🔴 重要: 以下のキー名が RealTimeGeminiProcessor.__init__ の引数名と1文字でも違うとTypeErrorになります
    kwargs={
        "whisper_model": load_whisper_model(),
        "api_key": api_key,
        "model_name": st.session_state.model_name,
        "terms": st.session_state.terms,
        "persona": persona
    },
)

output_area = st.empty()

if webrtc_ctx.state.playing:
    while True:
        try:
            # audio_workerが存在し、かつresult_queueを持っているか確認
            if hasattr(webrtc_ctx, 'audio_worker') and webrtc_ctx.audio_worker:
                new_line = webrtc_ctx.audio_worker.result_queue.get(timeout=1.0)
                st.session_state.full_notes += new_line + "\n\n"
                output_area.text_area("補正済みノート", value=st.session_state.full_notes, height=500)
            else:
                break
        except (queue.Empty, AttributeError):
            break
else:
    output_area.text_area("補正済みノート", value=st.session_state.full_notes, height=500)
