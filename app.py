import streamlit as st
import numpy as np
import queue
import docx
import google.generativeai as genai
from streamlit_webrtc import webrtc_streamer, WebRtcMode, AudioProcessorBase
from faster_whisper import WhisperModel
import torch

# --- 設定とモデルロード ---
st.set_page_config(page_title="リアルタイム専門用語補正ノート📝", layout="wide")

@st.cache_resource
def load_whisper_model():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # 授業用なので base モデルで速度を優先
    return WhisperModel("base", device=device, compute_type="float16" if device=="cuda" else "int8")

def extract_terms_with_gemini(file, api_key, model_name):
    doc = docx.Document(file)
    text = "\n".join([p.text for p in doc.paragraphs])
    context_text = text[:8000]

    genai.configure(api_key=api_key)
    # モデル名の指定を修正
    model = genai.GenerativeModel(model_name)
    
    prompt = f"以下の講義資料から、重要な専門用語や固有名詞を最大50個抜き出してください。カンマ区切りで単語のみを出力してください。\n\n【資料】:\n{context_text}"
    
    try:
        response = model.generate_content(prompt)
        terms = [t.strip() for t in response.text.split(",") if t.strip()]
        return terms
    except Exception as e:
        st.error(f"キーワード分析エラー: {e}")
        return ["法律", "政治", "国際社会"]

# --- 音声 & Gemini 処理クラス ---
class RealTimeGeminiProcessor(AudioProcessorBase):
    def __init__(self, whisper_model, api_key, model_name, terms, persona):
        self.whisper_model = whisper_model
        self.terms = terms
        self.persona = persona
        self.model_name = model_name
        self.audio_buffer = []
        self.result_queue = queue.Queue()
        
        # クラス内で個別に設定
        genai.configure(api_key=api_key)
        self.gemini_model = genai.GenerativeModel(model_name)

    def recv(self, frame):
        # AudioProcessorの標準的な実装
        audio = frame.to_ndarray().flatten().astype(np.float32) / 32768.0
        self.audio_buffer.extend(audio)

        if len(self.audio_buffer) >= 16000 * 3:
            segment_audio = np.array(self.audio_buffer)
            self.audio_buffer = []
            
            segments, _ = self.whisper_model.transcribe(
                segment_audio, 
                language="ja",
                initial_prompt=f"用語: {','.join(self.terms[:10])}"
            )
            raw_text = "".join([s.text for s in segments]).strip()

            if raw_text:
                prompt = f"あなたは{self.persona}です。以下の聞き取りを、用語リストを参考に修正してください。文章のみ出力。\n用語: {','.join(self.terms)}\n聞き取り: {text}"
                try:
                    # 直接モデルから生成
                    response = self.gemini_model.generate_content(
                        f"修正してください: {raw_text}\n用語ヒント: {','.join(self.terms)}"
                    )
                    self.result_queue.put(response.text.strip())
                except:
                    self.result_queue.put(raw_text)
        return frame

# --- UI ---
st.title("🎙️ リアルタイム講義補正ノート")

with st.sidebar:
    st.header("⚙️ 設定")
    api_key = st.text_input("Gemini API Key", type="password")
    # モデル名を内部用（models/...）に修正
    model_choice = st.selectbox("Gemini Model", ["models/gemini-1.5-flash", "models/gemini-1.5-pro"])
    persona = st.text_input("AIの役割", "東北大学法学部の教授")
    uploaded_docx = st.file_uploader("講義資料 (docx)", type="docx")

if not api_key or not uploaded_docx:
    st.info("設定を完了させてください。")
    st.stop()

if "terms" not in st.session_state:
    st.session_state.terms = extract_terms_with_gemini(uploaded_docx, api_key, model_choice)
if "full_notes" not in st.session_state:
    st.session_state.full_notes = ""

# WebRTC
webrtc_ctx = webrtc_streamer(
    key="lecture-gemini",
    mode=WebRtcMode.SENDONLY,
    media_stream_constraints={"video": False, "audio": True},
    rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
    worker_class=RealTimeGeminiProcessor,
    # 渡す値を__init__と完全に一致させる
    kwargs={
        "whisper_model": load_whisper_model(),
        "api_key": api_key,
        "model_name": model_choice,
        "terms": st.session_state.terms,
        "persona": persona
    },
)

output_area = st.empty()

if webrtc_ctx.state.playing:
    while True:
        try:
            # キューから結果を取得して表示
            new_line = webrtc_ctx.audio_worker.result_queue.get(timeout=1.0)
            st.session_state.full_notes += new_line + "\n\n"
            output_area.text_area("補正済みノート", value=st.session_state.full_notes, height=500)
        except (queue.Empty, AttributeError):
            break
else:
    output_area.text_area("補正済みノート", value=st.session_state.full_notes, height=500)
