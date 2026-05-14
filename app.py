import streamlit as st
import numpy as np
import queue
import docx
import MeCab
from collections import Counter
from streamlit_webrtc import webrtc_streamer, WebRtcMode, AudioProcessorBase
from faster_whisper import WhisperModel
import google.generativeai as genai
import torch

# --- 設定とモデルロード ---
st.set_page_config(page_title="リアルタイム専門用語補正ノート📝", layout="wide")

@st.cache_resource
def load_whisper_model():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    return WhisperModel("base", device=device, compute_type="float16" if device=="cuda" else "int8")
def extract_terms(file, top_n=100):
    doc = docx.Document(file)
    text = "\n".join([p.text for p in doc.paragraphs])
    
    # 辞書候補（環境によって異なるため、網羅的に指定）
    tagger = None
    paths = [
        "/var/lib/mecab/dic/ipadic-utf8",
        "/usr/lib/x86_64-linux-gnu/mecab/dic/ipadic",
        "/usr/lib/x86_64-linux-gnu/mecab/dic/ipadic-utf8",
        "/usr/share/mecab/dic/ipadic",
        "/etc/alternatives/mecab-dictionary"
    ]
    
    for p in paths:
        try:
            t = MeCab.Tagger(f"-d {p}")
            t.parse("") # 動作確認
            tagger = t
            break
        except:
            continue
            
    if tagger is None:
        # 最終手段：それでもダメな場合は、Dockerfileに以下の1行があるか確認してください
        # RUN apt-get install -y mecab mecab-ipadic-utf8
        tagger = MeCab.Tagger() 

    node = tagger.parseToNode(text)
    terms = []
    while node:
        features = node.feature.split(',')
        if features[0] == "名詞":
            word = node.surface
            if len(word) >= 3 or any(0x30A0 <= ord(c) <= 0x30FF for c in word):
                if len(word) >= 2:
                    terms.append(word)
        node = node.next
    return [term for term, count in Counter(terms).most_common(top_n)]

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
            segments, _ = self.whisper_model.transcribe(
                segment_audio, language="ja", initial_prompt=f"用語: {','.join(self.terms[:10])}"
            )
            raw_text = "".join([s.text for s in segments]).strip()
            if raw_text:
                corrected = self.correct_with_gemini(raw_text)
                self.result_queue.put(corrected)
        return frame

    def correct_with_gemini(self, text):
        prompt = f"あなたは{self.persona}です。以下の「聞き取り」を、用語リストを参考に修正してください。修正後の文章のみ出力すること。解説不要。\n用語リスト: {','.join(self.terms)}\n聞き取り: {text}"
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

if "terms" not in st.session_state:
    st.session_state.terms = extract_terms(uploaded_docx)
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
