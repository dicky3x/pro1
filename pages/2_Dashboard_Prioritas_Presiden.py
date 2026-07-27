"""
Halaman 2: Dashboard Prioritas Presiden
-------------------------------------------
Sumber data: prioritas_presiden_2026.xlsx (diolah via build_prioritas_strategis.py
menjadi data/prioritas_presiden.csv.gz). Baris ditandai prioritas berdasarkan kombinasi
kode Kegiatan, kode Output, dan kode Suboutput tertentu (sudah dipilih di file sumbernya).

Halaman ini TIDAK dibatasi ke satker milik user yang login (beda dgn Halaman 1) --
sama seperti Halaman 4, ini dashboard gambaran se-provinsi.
"""

import streamlit as st

from common import get_data, require_login, render_dashboard_kategori

st.set_page_config(
    page_title="DATUK - Dashboard Prioritas Presiden",
    page_icon="⭐",
    layout="wide",
)

df_login = get_data()
auth = require_login(df_login, judul_halaman="DATUK")

render_dashboard_kategori(
    data_path="data/prioritas_presiden.csv.gz",
    judul_halaman="Dashboard Prioritas Presiden",
    icon="⭐",
    label_kategori="Prioritas Presiden",
)
