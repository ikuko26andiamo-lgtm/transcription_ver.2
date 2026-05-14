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
    # 授業用なので base モデル。CPU環境でも比較的動きます
    return WhisperModel("base", device=device, compute_type="float16" if device=="cuda" else "int8")

def get_working_model(api_key):
    """利用可能なGeminiモデルを自動探索して、最初に使えるモデル名を返す"""
    genai.configure(api_key=api_key)
    # 試行するモデル候補の優先順位
    candidates = [
        "gemini-1.5-flash",
        "models/gemini-1.5-flash",
        "gemini-1.5-pro",
        "models/gemini-1.5-pro",
        "gemini-pro"
    ]
    
    available_models = []
    try:
        for m in genai.list_models():
            if 'generateContent' in m.supported_generation_methods:
                available_models.append(m.name)
        
        # 候補の中で、実際に利用可能リストにあるものを優先
        for cand in candidates:
            # candidateが models/ で始まっていない場合、比較用に整形
            full_cand = cand if cand.startswith("models/") else f"models/{cand}"
            if full_cand in available_models:
                return full_cand
        
        # 見つからなければリストの先頭を返す
        return available_models[0] if available_models else "gemini-1.5-flash"
    except:
        # リスト取得に失敗した場合はデフォルトを返す
        return "gemini-1.5-flash"

def extract_terms_with_gemini(file, api_key, model_name):
    doc = docx.Document(file)
    text = "\n".join([p.text for p in doc.paragraphs])
    context_text = text[:8000]

    genai.configure(api_key=api_key)
    model = genai.GenerativeModel(model_name)
    
    prompt = f"以下の講義資料から、重要な専門用語や固有名詞を最大50個抜き出してください。カンマ区切りで単語のみを出力してください。\n\n【資料】:\n{context_text}"
    
    try:
        response = model.generate_content(prompt)
        terms = [t.strip() for t in response.text.split(",") if t.strip()]
        return terms
    except Exception as e:
        # ここで失敗しても動作を止めない
        return ["法律", "政治", "国際社会", "裁判所", "憲法"]

# --- 音声 & Gemini 処理クラス ---
class RealTimeGeminiProcessor(AudioProcessorBase):
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
                    segment_audio, 
                    language="ja",
                    initial_prompt=f"専門用語: {','.join(self.terms[:10])}"
                )
                raw_text = "".join([s.text for s in segments]).strip()

                if raw_text:
                    # 補正プロンプト
                    p = f"修正して: {raw_text}\nヒント: {','.join(self.terms[:30])}"
                    response = self.gemini_model.generate_content(p)
                    self.result_queue.put(response.text.strip())
            except Exception:
                # エラー時は何もしないか、生テキストを流す
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

# モデルの自動決定
if "model_name" not in st.session_state:
    with st.spinner("接続可能なAIモデルを探索中..."):
        st.session_state.model_name = get_working_model(api_key)
    st.success(f"使用モデル: {st.session_state.model_name}")

if "terms" not in st.session_state:
    with st.spinner("資料を分析中..."):
        st.session_state.terms = extract_terms_with_gemini(uploaded_docx, api_key, st.session_state.model_name)

if "full_notes" not in st.session_state:
    st.session_state.full_notes = ""

# WebRTC (kwargsをProcessorの__init__と完全に一致させる)
webrtc_ctx = webrtc_streamer(
    key="lecture-gemini",
    mode=WebRtcMode.SENDONLY,
    media_stream_constraints={"video": False, "audio": True},
    rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
    worker_class=RealTimeGeminiProcessor,
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
            # 属性エラーを回避するため、workerが存在するか確認
            if webrtc_ctx.audio_worker:
                new_line = webrtc_ctx.audio_worker.result_queue.get(timeout=1.0)
                st.session_state.full_notes += new_line + "\n\n"
                output_area.text_area("補正済みノート", value=st.session_state.full_notes, height=500)
            else:
                break
        except (queue.Empty, AttributeError):
            break
else:
    output_area.text_area("補正済みノート", value=st.session_state.full_notes, height=500)
