"""
Halaman 3: Dashboard Program Strategis
-------------------------------------------
Sumber data: program_strategis2026.xlsx (diolah via build_prioritas_strategis.py
menjadi data/program_strategis.csv.gz). Baris ditandai strategis berdasarkan kombinasi
kode Kegiatan, kode Output, dan kode Suboutput tertentu (sudah dipilih di file sumbernya).

Akses: halaman ini otomatis disembunyikan dari navigasi (lihat app.py) kalau satker yang
login tidak punya anggaran terkait program strategis. Kalau muncul, datanya dibatasi ke
satker itu sendiri saja -- kecuali super user, yang melihat gambaran se-provinsi.
"""

import streamlit as st

from common import render_dashboard_kategori

# Login sudah ditangani app.py (router) sebelum halaman ini dipanggil -- auth dijamin ada.
auth = st.session_state.auth
is_super = auth["role"] == "super"
scope_kdsatker = None if is_super else auth["kdsatker"]

render_dashboard_kategori(
    data_path="data/program_strategis.csv.gz",
    judul_halaman="Dashboard Program Strategis",
    icon="🎯",
    label_kategori="Program Strategis",
    scope_kdsatker=scope_kdsatker,
    page_key="strategis",
)
