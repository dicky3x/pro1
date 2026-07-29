"""
Halaman 2: Dashboard Prioritas Presiden
-------------------------------------------
Sumber data: prioritas_presiden_2026.xlsx (diolah via build_prioritas_strategis.py
menjadi data/prioritas_presiden.csv.gz). Baris ditandai prioritas berdasarkan kombinasi
kode Kegiatan, kode Output, dan kode Suboutput tertentu (sudah dipilih di file sumbernya).

Akses: halaman ini otomatis disembunyikan dari navigasi (lihat app.py) kalau satker yang
login tidak punya anggaran terkait prioritas presiden. Kalau muncul, datanya dibatasi ke
satker itu sendiri saja -- kecuali super user, yang melihat gambaran se-provinsi.
"""

import streamlit as st

from common import render_dashboard_kategori

# Login sudah ditangani app.py (router) sebelum halaman ini dipanggil -- auth dijamin ada.
auth = st.session_state.auth
is_super = auth["role"] == "super"
scope_kdsatker = None if is_super else auth["kdsatker"]

render_dashboard_kategori(
    data_path="data/prioritas_presiden.csv.gz",
    judul_halaman="Dashboard Prioritas Presiden",
    icon="⭐",
    label_kategori="Prioritas Presiden",
    scope_kdsatker=scope_kdsatker,
    page_key="prioritas",
)
