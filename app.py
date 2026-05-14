import streamlit as st
import numpy as np
import queue
import docx
import google.generativeai as genai
from streamlit_webrtc import webrtc_streamer, WebRtcMode, AudioProcessorBase
from faster_whisper import WhisperModel
import torch
import threading
from collections import Counter

# --- 1. 基本設定とSession State初期化 ---
st.set_page_config(page_title="リアルタイム講義補正ノート📝", layout="wide")

if "model_name" not in st.session_state:
    st.session_state.model_name = "models/gemini-1.5-flash"
if "terms" not in st.session_state:
    st.session_state.terms = []
if "full_notes" not in st.session_state:
    st.session_state.full_notes = ""

# --- 2. モデル・データ処理関数 ---
@st.cache_resource
def load_whisper_model():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # CPU環境でも動くよう base モデルを採用
    return WhisperModel("base", device=device, compute_type="float16" if device=="cuda" else "int8")

def get_working_model(api_key):
    try:
        genai.configure(api_key=api_key)
        available_models = [m.name for m in genai.list_models() if 'generateContent' in m.supported_generation_methods]
        candidates = ["models/gemini-1.5-flash", "models/gemini-1.5-pro", "models/gemini-pro"]
        for cand in candidates:
            if cand in available_models: return cand
        return available_models[0] if available_models else "models/gemini-1.5-flash"
    except:
        return "models/gemini-1.5-flash"

def extract_terms_with_gemini(file, api_key, model_name):
    doc = docx.Document(file)
    text = "\n".join([p.text for p in doc.paragraphs])
    context_text = text[:8000]
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel(model_name)
    prompt = f"以下の講義資料から、文字起こし補正に役立つ専門用語を最大50個抜き出し、カンマ区切りで単語のみ出力してください。\n\n{context_text}"
    try:
        response = model.generate_content(prompt)
        return [t.strip() for t in response.text.split(",") if t.strip()]
    except:
        return ["法律", "政治", "国際社会", "憲法", "主権"]

# --- 3. 音声処理クラス（スレッド分離 & リサンプリング対応） ---
class RealTimeGeminiProcessor(AudioProcessorBase):
    def __init__(self, whisper_model, api_key, model_name, terms, persona):
        self.whisper_model = whisper_model
        self.terms = terms
        self.persona = persona
        self.audio_queue = queue.Queue()
        self.result_queue = queue.Queue()
        genai.configure(api_key=api_key)
        self.gemini_model = genai.GenerativeModel(model_name)
        
        # 解析用スレッドを起動
        self.processing_thread = threading.Thread(target=self._process_loop, daemon=True)
        self.processing_thread.start()

    def recv(self, frame):
        # 音声受け取り
        audio = frame.to_ndarray()
        if audio.ndim > 1:
            audio = np.mean(audio, axis=1)
        
        # 48kHzなどから16kHzへリサンプリング
        sample_rate = frame.sample_rate
        if sample_rate != 16000:
            step = sample_rate // 16000
            audio = audio[::step]

        self.audio_queue.put(audio.flatten().astype(np.float32) / 32768.0)
        return frame

    def _process_loop(self):
        buffer = []
        while True:
            chunk = self.audio_queue.get()
            buffer.extend(chunk)

            # 5秒溜まったらAI解析
            if len(buffer) >= 16000 * 5:
                segment_audio = np.array(buffer)
                buffer = []
                try:
                    segments, _ = self.whisper_model.transcribe(
                        segment_audio, language="ja",
                        initial_prompt=f"専門用語: {','.join(self.terms[:10])}"
                    )
                    raw_text = "".join([s.text for s in segments]).strip()
                    if raw_text:
                        p = f"あなたは{self.persona}です。以下の文を用語リストを参考に修正してください。修正後の文のみ出力。\n用語: {','.join(self.terms[:25])}\n文: {raw_text}"
                        response = self.gemini_model.generate_content(p)
                        self.result_queue.put(response.text.strip())
                except:
                    pass

# --- 4. メイン UI ---
st.title("🎙️ リアルタイム講義補正ノート")

with st.sidebar:
    st.header("⚙️ 設定")
    api_key = st.text_input("Gemini API Key", type="password")
    persona = st.text_input("AIの役割", "法学部の教授")
    uploaded_docx = st.file_uploader("講義資料 (docx)", type="docx")
    
    if st.button("📋 ノートをリセット"):
        st.session_state.full_notes = ""
        st.rerun()

# 入力待ちガード
if not api_key or not uploaded_docx:
    st.info("APIキーの入力と講義資料をアップロードしてください。")
    st.stop()

# データ確定処理
if not st.session_state.terms:
    with st.spinner("資料から専門用語を分析中..."):
        st.session_state.model_name = get_working_model(api_key)
        st.session_state.terms = extract_terms_with_gemini(uploaded_docx, api_key, st.session_state.model_name)
    st.success(f"分析完了！使用モデル: {st.session_state.model_name}")

# WebRTC用変数のキャプチャ
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

# --- 5. WebRTCコンポーネント ---
webrtc_ctx = webrtc_streamer(
    key="lecture-gemini",
    mode=WebRtcMode.SENDONLY,
    media_stream_constraints={
        "video": False, 
        "audio": {
            "echoCancellation": True,
            "noiseSuppression": True,
        }
    },
    rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
    audio_processor_factory=audio_processor_factory,
    async_processing=True,
)

# --- 6. ノート表示エリア ---
st.subheader("📝 リアルタイム補正結果")
output_area = st.empty()

if webrtc_ctx.state.playing:
    while True:
        try:
            if hasattr(webrtc_ctx, 'audio_processor') and webrtc_ctx.audio_processor:
                new_line = webrtc_ctx.audio_processor.result_queue.get(timeout=1.0)
                st.session_state.full_notes += new_line + "\n\n"
                output_area.text_area("講義ノート", value=st.session_state.full_notes, height=500)
            else:
                break
        except (queue.Empty, AttributeError):
            break
else:
    output_area.text_area("講義ノート", value=st.session_state.full_notes, height=500)
