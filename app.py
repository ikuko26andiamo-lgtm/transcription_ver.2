import streamlit as st
import numpy as np
import queue
import docx
import google.generativeai as genai
from streamlit_webrtc import webrtc_streamer, WebRtcMode, AudioProcessorBase
from faster_whisper import WhisperModel
import torch

# --- 1. 基本設定 ---
st.set_page_config(page_title="リアルタイム講義補正ノート📝", layout="wide")

@st.cache_resource
def load_whisper_model():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    return WhisperModel("base", device=device, compute_type="float16" if device=="cuda" else "int8")

def get_working_model(api_key):
    try:
        genai.configure(api_key=api_key)
        available_models = [m.name for m in genai.list_models() if 'generateContent' in m.supported_generation_methods]
        candidates = ["models/gemini-1.5-flash", "models/gemini-1.5-pro"]
        for cand in candidates:
            if cand in available_models: return cand
        return "models/gemini-1.5-flash"
    except:
        return "models/gemini-1.5-flash"

def extract_terms_with_gemini(file, api_key, model_name):
    doc = docx.Document(file)
    text = "\n".join([p.text for p in doc.paragraphs])
    context_text = text[:8000]
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel(model_name)
    prompt = f"以下の資料から重要な専門用語を最大50個抜き出し、カンマ区切りで単語のみ出力してください。\n\n{context_text}"
    try:
        response = model.generate_content(prompt)
        return [t.strip() for t in response.text.split(",") if t.strip()]
    except:
        return ["法律", "政治", "憲法"]

# --- 2. 音声処理クラス ---
class RealTimeGeminiProcessor(AudioProcessorBase):
    def __init__(self, whisper_model, api_key, model_name, terms, persona):
        self.whisper_model = whisper_model
        self.terms = terms
        self.persona = persona
        self.audio_buffer = []
        self.result_queue = queue.Queue()
        genai.configure(api_key=api_key)
        self.gemini_model = genai.GenerativeModel(model_name)

    def recv(self, frame):
        # 1. 音声データをndarrayとして取得
        audio = frame.to_ndarray()
        
        # 2. ステレオ(2ch)をモノラル(1ch)に変換
        if audio.ndim > 1:
            audio = np.mean(audio, axis=1)
        
        # 3. ブラウザの周波数(多くは48kHz)をWhisper用の16kHzに変換
        # 48000Hz -> 16000Hz なので 3サンプルに1つ間引く
        sample_rate = frame.sample_rate
        if sample_rate != 16000:
            # 簡易的なダウンサンプリング処理
            step = sample_rate // 16000
            audio = audio[::step]
            
        # 4. 数値を正規化してバッファへ
        audio = audio.flatten().astype(np.float32) / 32768.0
        self.audio_buffer.extend(audio)

        # 5秒分(16000 * 5 = 80000サンプル)溜まったら処理
        if len(self.audio_buffer) >= 80000:
            segment_audio = np.array(self.audio_buffer)
            self.audio_buffer = []
            
            try:
                # Whisper推論
                segments, _ = self.whisper_model.transcribe(
                    segment_audio, 
                    language="ja",
                    beam_size=5, # 精度を上げる
                    initial_prompt=f"専門用語: {','.join(self.terms[:10])}"
                )
                raw_text = "".join([s.text for s in segments]).strip()
                
                if raw_text:
                    # Gemini補正
                    p = f"修正して: {raw_text}\n用語: {','.join(self.terms[:20])}"
                    response = self.gemini_model.generate_content(p)
                    self.result_queue.put(response.text.strip())
                else:
                    # デバッグ：音が届いていることを確認するために、空でもキューを送る設定にしてみる
                    # self.result_queue.put("（音声なし）")
                    pass
            except Exception as e:
                self.result_queue.put(f"⚠️ 解析エラー")
        
        return frame
       
# --- 3. メイン UI ---
st.title("🎙️ リアルタイム講義補正ノート")

with st.sidebar:
    st.header("⚙️ 設定")
    api_key = st.text_input("Gemini API Key", type="password")
    persona = st.text_input("AIの役割", "法学部の教授")
    uploaded_docx = st.file_uploader("講義資料 (docx)", type="docx")

if not api_key or not uploaded_docx:
    st.info("APIキーと資料をセットしてください。")
    st.stop()

if "model_name" not in st.session_state:
    st.session_state.model_name = get_working_model(api_key)
if "terms" not in st.session_state:
    st.session_state.terms = extract_terms_with_gemini(uploaded_docx, api_key, st.session_state.model_name)
if "full_notes" not in st.session_state:
    st.session_state.full_notes = ""

# --- 4. WebRTC (スレッドセーフな変数渡し) ---
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
)

# --- 5. 画面表示 ---
st.subheader("📝 補正済みノート")
output_area = st.empty()

if webrtc_ctx.state.playing:
    while True:
        try:
            if webrtc_ctx.audio_processor:
                new_line = webrtc_ctx.audio_processor.result_queue.get(timeout=1.0)
                st.session_state.full_notes += new_line + "\n\n"
                output_area.text_area("テキスト", value=st.session_state.full_notes, height=500)
            else:
                break
        except (queue.Empty, AttributeError):
            break
else:
    output_area.text_area("テキスト", value=st.session_state.full_notes, height=500)
