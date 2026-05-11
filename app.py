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
st.set_page_config(page_title="リアルタイム文字起こし📝", layout="wide")

@st.cache_resource
def load_whisper_model():
    # リアルタイム性重視のため "base" モデルを使用
    device = "cuda" if torch.cuda.is_available() else "cpu"
    return WhisperModel("base", device=device, compute_type="float16" if device=="cuda" else "int8")

def extract_terms(file):
    doc = docx.Document(file)
    text = "\n".join([p.text for p in doc.paragraphs])
    tagger = MeCab.Tagger()
    node = tagger.parseToNode(text)
    terms = []
    while node:
        features = node.feature.split(',')
        if features[0] == "名詞":
            word = node.surface
            if len(word) >= 2: terms.append(word)
        node = node.next
    return [t for t, _ in Counter(terms).most_common(100)]

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
        # Geminiの設定
        genai.configure(api_key=self.api_key)
        self.gemini_model = genai.GenerativeModel(self.model_name)

    def recv(self, frame):
        audio = frame.to_ndarray().flatten().astype(np.float32) / 32768.0
        self.audio_buffer.extend(audio)

        # 3秒（16000Hz * 3）溜まったら処理
        if len(self.audio_buffer) >= 48000:
            segment_audio = np.array(self.audio_buffer)
            self.audio_buffer = []

            # 1. Whisperで文字起こし
            segments, _ = self.whisper_model.transcribe(segment_audio, language="ja")
            raw_text = "".join([s.text for s in segments]).strip()

            if raw_text:
                # 2. Geminiで専門用語補正
                corrected = self.correct_text(raw_text)
                self.result_queue.put(corrected)
        return frame

    def correct_text(self, text):
        prompt = f"""
あなたは{self.persona}です。以下の「聞き取り」を、用語リストを参考に正しく修正してください。
文脈から判断し、特に人名や専門用語の誤字を直してください。
出力は「修正後の文章のみ」とし、解説は一切不要です。

用語リスト: {",".join(self.terms[:50])}
聞き取り: {text}
"""
        try:
            response = self.gemini_model.generate_content(prompt)
            return response.text.strip()
        except:
            return f"(修正失敗) {text}"

# --- メイン UI ---
st.title("🎓 リアルタイム専門用語補正ノート")

with st.sidebar:
    st.header("⚙️ 設定")
    api_key = st.text_input("Gemini API Key", type="password")
    selected_model = st.selectbox("Gemini Model", ["gemini-1.5-flash", "gemini-1.5-pro"])
    persona = st.text_input("専門家設定", "法学部・政治史の教授")
    uploaded_docx = st.file_uploader("講義資料 (docx)", type="docx")

if not api_key or not uploaded_docx:
    st.warning("APIキーと講義資料をセットしてください。")
    st.stop()

# 事前準備
if "terms" not in st.session_state:
    st.session_state.terms = extract_terms(uploaded_docx)
if "final_text" not in st.session_state:
    st.session_state.full_notes = ""

st.write(f"✅ 抽出された主要用語: {', '.join(st.session_state.terms[:10])}...")

# リアルタイム処理実行
webrtc_ctx = webrtc_streamer(
    key="gemini-lecture",
    mode=WebRtcMode.SENDONLY,
    audio_receiver_size=1024,
    media_stream_constraints={"video": False, "audio": True},
    worker_class=RealTimeGeminiProcessor,
    kwargs={
        "whisper_model": load_whisper_model(),
        "api_key": api_key,
        "model_name": selected_model,
        "terms": st.session_state.terms,
        "persona": persona
    },
)

# 表示エリア
st.subheader("📝 リアルタイム文字起こし")
output_area = st.empty()

if webrtc_ctx.state.playing:
    while True:
        try:
            new_line = webrtc_ctx.audio_worker.result_queue.get(timeout=1.0)
            st.session_state.full_notes += new_line + "\n\n"
            output_area.text_area("補正済みテキスト", value=st.session_state.full_notes, height=500)
        except queue.Empty:
            break            # 3文字以上、またはカタカナ（専門用語に多い）を優先
            if len(word) >= 3 or any(0x30A0 <= ord(c) <= 0x30FF for c in word):
                if len(word) >= 2:
                    terms.append(word)
        node = node.next

    return [term for term, count in Counter(terms).most_common(top_n)]


class RealTimeLectureProcessor(AudioProcessorBase):
    def __init__(self, whisper_model, api_key, model_name, terms, persona):
        self.whisper_model = whisper_model
        self.api_key = api_key
        self.model_name = model_name
        self.terms = terms
        self.persona = persona
        self.audio_buffer = []
        self.result_queue = queue.Queue()

    def recv(self, frame):
        # 16kHz, float32に変換してバッファへ
        audio = frame.to_ndarray().flatten().astype(np.float32) / 32768.0
        self.audio_buffer.extend(audio)

        # 約3秒分（48000サンプル）溜まったら処理
        if len(self.audio_buffer) >= 16000 * 3:
            segment_audio = np.array(self.audio_buffer)
            self.audio_buffer = []

            # 1. Whisperで一旦文字起こし
            segments, _ = self.whisper_model.transcribe(
                segment_audio, language="ja", initial_prompt=f"用語: {','.join(self.terms[:10])}"
            )
            raw_text = "".join([s.text for s in segments]).strip()

            if raw_text:
                # 2. 即座にGeminiに投げて専門用語を修正
                corrected_text = self.fix_with_gemini(raw_text)
                self.result_queue.put(corrected_text)

        return frame

    def fix_with_gemini(self, text):
        # 3秒ごとの短いテキストを修正するための軽量プロンプト
        prompt = f"""
あなたは{self.persona}です。以下の「聞き取り間違いを含む可能性のあるテキスト」を、
提供された「用語リスト」を参考にして、正しい専門用語に置き換えて1行で出力してください。
余計な解説などは追加してはいけません。

【用語リスト】: {','.join(self.terms)}
【テキスト】: {text}
"""
        # ここで genai.GenerativeModel を呼び出す（既存の call_gemini を軽量化したもの）
        import google.generativeai as genai
        genai.configure(api_key=self.api_key)
        model = genai.GenerativeModel(self.model_name)
        try:
            response = model.generate_content(prompt)
            return response.text.strip()
        except:
            return text # エラー時は原文を返す

# --- UI (メインループ) ---
# ※webrtc_streamerのkwargsに、api_keyやtermsを渡して初期化します
