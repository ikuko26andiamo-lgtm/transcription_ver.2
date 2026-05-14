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
    return WhisperModel("base", device=device, compute_type="float16" if device=="cuda" else "int8")

# MeCabの代わりにGeminiを使ってキーワードを抽出する
def extract_terms_with_gemini(file, api_key, model_name):
    doc = docx.Document(file)
    text = "\n".join([p.text for p in doc.paragraphs])
    
    # 資料が長すぎる場合、前半1万文字程度を対象にする
    context_text = text[:10000]

    genai.configure(api_key=api_key)
    model = genai.GenerativeModel(model_name)
    
    prompt = f"""
以下の講義資料から、リアルタイム文字起こしの補正に役立つ「専門用語・人名・固有名詞」を最大60個、重要度順に抜き出してください。
余計な解説は不要です。カンマ区切りで単語のみを出力してください。

【資料】:
{context_text}
"""
    try:
        response = model.generate_content(prompt)
        # カンマ区切りの文字列をリスト化
        terms = [t.strip() for t in response.text.split(",") if t.strip()]
        return terms
    except Exception as e:
        st.error(f"キーワード分析に失敗しました: {e}")
        return ["法律", "政治"] # 最低限のフォールバック

# --- 音声 & Gemini 処理クラス ---
class RealTimeGeminiProcessor(AudioProcessorBase):
    def __init__(self, whisper_model, api_key, model_name, terms, persona):
        self.whisper_model = whisper_model
        self.api_key = api_key
        self.model_name = model_name
        self.terms = terms
        self.persona = persona
        self.audio_buffer = []
        self.result_queue = queue.Queue()
        genai.configure(api_key=self.api_key)
        self.gemini_model = genai.GenerativeModel(self.model_name)

    def recv(self, frame):
        audio = frame.to_ndarray().flatten().astype(np.float32) / 32768.0
        self.audio_buffer.extend(audio)
        if len(self.audio_buffer) >= 16000 * 3:
            segment_audio = np.array(self.audio_buffer)
            self.audio_buffer = []
            
            # Whisperで仮起こし
            segments, _ = self.whisper_model.transcribe(
                segment_audio, 
                language="ja",
                initial_prompt=f"専門用語: {','.join(self.terms[:10])}"
            )
            raw_text = "".join([s.text for s in segments]).strip()

            if raw_text:
                # Geminiで補正
                corrected = self.correct_with_gemini(raw_text)
                self.result_queue.put(corrected)
        return frame

    def correct_with_gemini(self, text):
        prompt = f"あなたは{self.persona}です。以下の「聞き取り」を用語リストを参考に正しく修正してください。修正後の文章のみ出力。解説不要。\n用語: {','.join(self.terms)}\n聞き取り: {text}"
        try:
            response = self.gemini_model.generate_content(prompt)
            return response.text.strip()
        except:
            return text 

# --- UI ---
st.title("🎓 リアルタイム専門用語補正ノート")

with st.sidebar:
    st.header("⚙️ 設定")
    api_key = st.text_input("Gemini API Key", type="password")
    selected_model = st.selectbox("Gemini Model", ["gemini-1.5-flash", "gemini-1.5-pro"])
    persona = st.text_input("専門家設定", "法学部・国際政治学の教授")
    uploaded_docx = st.file_uploader("講義参考資料 (docx)", type="docx")

if not api_key or not uploaded_docx:
    st.info("APIキーの入力と資料のアップロードを行ってください。")
    st.stop()

# セッション管理
if "terms" not in st.session_state:
    with st.spinner("Geminiが講義資料を分析中..."):
        st.session_state.terms = extract_terms_with_gemini(uploaded_docx, api_key, selected_model)
if "full_notes" not in st.session_state:
    st.session_state.full_notes = ""

st.success(f"キーワード抽出完了: {', '.join(st.session_state.terms[:10])}...")

webrtc_ctx = webrtc_streamer(
    key="lecture-gemini",
    mode=WebRtcMode.SENDONLY,
    media_stream_constraints={"video": False, "audio": True},
    rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
    worker_class=RealTimeGeminiProcessor,
    kwargs={
        "whisper_model": load_whisper_model(),
        "api_key": api_key,
        "model_name": selected_model,
        "terms": st.session_state.terms,
        "persona": persona
    },
)

output_area = st.empty()

if webrtc_ctx.state.playing:
    while True:
        try:
            new_line = webrtc_ctx.audio_worker.result_queue.get(timeout=1.0)
            st.session_state.full_notes += new_line + "\n\n"
            output_area.text_area("ノート", value=st.session_state.full_notes, height=500)
        except queue.Empty:
            break
else:
    output_area.text_area("ノート", value=st.session_state.full_notes, height=500)

if st.button("📋 リセット"):
    st.session_state.full_notes = ""
    st.rerun()
