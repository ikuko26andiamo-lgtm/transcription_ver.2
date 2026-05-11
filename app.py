import streamlit as st
from streamlit_webrtc import webrtc_streamer, WebRtcMode, AudioProcessorBase
import queue
import numpy as np

# --- ページ設定 ---
st.set_page_config(page_title="文字起こし自動作成", page_icon="📝")

# --- モデル探索機能の追加 ---
def get_available_gemini_models(api_key):
    """利用可能なGeminiモデルをリストアップする"""
    try:
        genai.configure(api_key=api_key)
        models = []
        for m in genai.list_models():
            if 'generateContent' in m.supported_generation_methods:
                # 'models/' のプレフィックスを取り除いて表示用にする
                models.append(m.name.replace('models/', ''))
        return models
    except Exception as e:
        st.error(f"モデルの取得に失敗しました: {e}")
        return ["gemini-1.5-flash", "gemini-1.5-pro", "gemini-pro"]

# キーワード抽出
def extract_terms(file, top_n=60):
    doc = docx.Document(file)
    text = "\n".join([p.text for p in doc.paragraphs])

    tagger = MeCab.Tagger()
    node = tagger.parseToNode(text)

    terms = []
    while node:
        features = node.feature.split(',')
        if features[0] == "名詞":
            word = node.surface
            # 3文字以上、またはカタカナ（専門用語に多い）を優先
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
