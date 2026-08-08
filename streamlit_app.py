"""
streamlit_app.py — Zafer Hoca asistani (Streamlit surumu).

Yerelde: streamlit run streamlit_app.py
Bulutta: GitHub reposuna koy -> share.streamlit.io'dan deploy et.

Notlar:
- API anahtari once st.secrets'tan, yoksa .env'den okunur.
- Sohbet gecmisi oturum bazlidir: her kullanicinin kendi gecmisi olur.
- Streamlit Cloud'un eski sqlite'i icin pysqlite3 yamasi en ustte olmali.
"""

# --- ChromaDB / Streamlit Cloud sqlite yamasi (ilk satirlar olmali) ---------
try:
    __import__("pysqlite3")
    import sys
    sys.modules["sqlite3"] = sys.modules.pop("pysqlite3")
except ImportError:
    pass  # yerelde pysqlite3 yoksa sistem sqlite'i yeterli

import os
import pathlib
import tempfile

import streamlit as st

# Anahtari core'dan ONCE ortama koy: bulutta st.secrets, yerelde .env kullanilir.
# Yerelde secrets.toml yoksa st.secrets'a dokunmak hata firlatir — o yuzden sarili.
try:
    _secrets = dict(st.secrets)
except Exception:
    _secrets = {}

for opt in ("ANTHROPIC_API_KEY", "ZAFER_LLM_MODEL", "ZAFER_EMBED_MODEL", "ZAFER_DB_PATH"):
    if opt in _secrets:
        os.environ[opt] = str(_secrets[opt])

from zafer_core import ZaferHoca, read_any, SUPPORTED  # noqa: E402

st.set_page_config(page_title="Zafer Hoca Asistani", page_icon="🎓", layout="centered")
st.title("Zafer Hoca Asistani")
st.caption("Ders materyallerine dayali veri bilimi danismani")

# --- Oturum durumu ----------------------------------------------------------
if "hoca" not in st.session_state:
    st.session_state.hoca = ZaferHoca()
if "messages" not in st.session_state:
    st.session_state.messages = []

# --- Kenar cubugu -----------------------------------------------------------
with st.sidebar:
    st.subheader("Dosya ekle")
    uploads = st.file_uploader(
        "Sonraki soruna eklenecek dosyalar",
        type=[s.lstrip(".") for s in SUPPORTED],
        accept_multiple_files=True,
    )
    st.divider()
    if st.button("Yeni proje", use_container_width=True):
        st.session_state.hoca.reset()
        st.session_state.messages = []
        st.rerun()
    st.caption(
        "Yeni proje: sohbet gecmisini temizler. "
        "Konu degistirirken kullan."
    )

# --- Gecmisi ciz ------------------------------------------------------------
for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])

# --- Girdi ------------------------------------------------------------------
prompt = st.chat_input("Sorunu yaz...")

if prompt:
    # Ekli dosyalari oku
    attached_text, names = "", []
    if uploads:
        blobs = []
        for up in uploads:
            suffix = pathlib.Path(up.name).suffix
            with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
                tmp.write(up.getvalue())
                tmp_path = pathlib.Path(tmp.name)
            text = read_any(tmp_path)
            tmp_path.unlink(missing_ok=True)
            if text:
                blobs.append(f"===== {up.name} =====\n{text}")
                names.append(up.name)
        attached_text = "\n\n".join(blobs)

    shown = prompt if not names else f"{prompt}\n\n_[Ekli: {', '.join(names)}]_"
    st.session_state.messages.append({"role": "user", "content": shown})
    with st.chat_message("user"):
        st.markdown(shown)

    with st.chat_message("assistant"):
        with st.spinner("Materyaller taraniyor..."):
            answer = st.session_state.hoca.ask(prompt, attached_text or None)
            st.markdown(answer)
    st.session_state.messages.append({"role": "assistant", "content": answer})
