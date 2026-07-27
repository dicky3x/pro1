"""
Halaman 3: Dashboard Program Strategis
-------------------------------------------
Sumber data: program_strategis2026.xlsx (diolah via build_prioritas_strategis.py
menjadi data/program_strategis.csv.gz). Baris ditandai strategis berdasarkan kombinasi
kode Kegiatan, kode Output, dan kode Suboutput tertentu (sudah dipilih di file sumbernya).

Halaman ini TIDAK dibatasi ke satker milik user yang login (beda dgn Halaman 1) --
sama seperti Halaman 4, ini dashboard gambaran se-provinsi.
"""

import streamlit as st

from common import get_data, require_login, render_dashboard_kategori

st.set_page_config(
    page_title="DATUK - Dashboard Program Strategis",
    page_icon="🎯",
    layout="wide",
)

df_login = get_data()
auth = require_login(df_login, judul_halaman="DATUK")

render_dashboard_kategori(
    data_path="data/program_strategis.csv.gz",
    judul_halaman="Dashboard Program Strategis",
    icon="🎯",
    label_kategori="Program Strategis",
)
