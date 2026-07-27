"""
Dashboard Multi-Halaman (Pagu & Realisasi, Program Prioritas, TKD)
------------------------------------------------------------------
Streamlit + Groq (narasi & chat AI). 
Menggunakan sidebar radio button untuk navigasi antar halaman dalam 1 file.
"""

import os
from datetime import date

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from groq import Groq

# --------------------------------------------------------------------------
# Konfigurasi dasar
# --------------------------------------------------------------------------

st.set_page_config(
    page_title="Dashboard Anggaran dan Transfer Riau Terkini",
    page_icon="📊",
    layout="wide",
)

BULAN_KOLOM = ["JAN", "FEB", "MAR", "APR", "MEI", "JUN",
               "JUL", "AGS", "SEP", "OKT", "NOV", "DES"]
BULAN_LABEL = {i + 1: b for i, b in enumerate(BULAN_KOLOM)}

LABEL_JENIS_BELANJA_SINGKAT = {
    51: "Belanja Pegawai",
    52: "Belanja Barang",
    53: "Belanja Modal",
    54: "Belanja Bunga Utang",
    55: "Belanja Subsidi",
    56: "Belanja Hibah",
    57: "Belanja Bansos",
    58: "Belanja Lain-lain",
    61: "DBH",
    62: "DAU",
    63: "DAK Fisik",
    64: "Insentif Daerah",
    65: "DAK Nonfisik",
    66: "Dana Desa",
}

GROQ_MODEL = os.environ.get("GROQ_MODEL", "openai/gpt-oss-120b")

def fmt_satker(kode) -> str:
    if kode is None:
        return ""
    try:
        return f"{int(kode):06d}"
    except (TypeError, ValueError):
        return str(kode)

def fmt_dept(kode) -> str:
    if kode is None:
        return ""
    try:
        return f"{int(kode):03d}"
    except (TypeError, ValueError):
        return str(kode)

# --------------------------------------------------------------------------
# Load data utama (Untuk Halaman 1 & 3)
# --------------------------------------------------------------------------

@st.cache_data(show_spinner="Memuat data utama...")
def load_data_from_csv(path: str) -> pd.DataFrame:
    try:
        return pd.read_csv(path)
    except FileNotFoundError:
        # Fallback dummy jika file tidak ditemukan saat pengembangan
        st.error(f"File {path} tidak ditemukan. Memuat data kosong.")
        return pd.DataFrame(columns=["TAHUN", "KDDEPT", "NMDEPT", "KDSATKER", "NMSATKER", "PAGU", "JENIS BELANJA", "BLOKIR"] + BULAN_KOLOM)

@st.cache_data(show_spinner="Memuat data dari Supabase...")
def load_data_from_supabase(url: str, key: str, table: str) -> pd.DataFrame:
    from supabase import create_client
    client = create_client(url, key)
    all_rows, page, page_size = [], 0, 1000
    while True:
        resp = client.table(table).select("*").range(page * page_size, (page + 1) * page_size - 1).execute()
        rows = resp.data
        all_rows.extend(rows)
        if len(rows) < page_size: break
        page += 1
    return pd.DataFrame(all_rows)

def load_data() -> pd.DataFrame:
    use_supabase = st.secrets.get("USE_SUPABASE", "false") == "true" if hasattr(st, "secrets") else False
    if use_supabase:
        return load_data_from_supabase(st.secrets["SUPABASE_URL"], st.secrets["SUPABASE_KEY"], st.secrets.get("SUPABASE_TABLE", "pagu_realisasi"))
    return load_data_from_csv("data/pagu_realisasi.csv.gz")

@st.cache_data(show_spinner=False)
def tanggal_update_data(path_csv: str = "data/pagu_realisasi.csv.gz") -> str:
    import subprocess
    from datetime import datetime as _dt
    nama_bulan = ["Januari", "Februari", "Maret", "April", "Mei", "Juni", "Juli", "Agustus", "September", "Oktober", "November", "Desember"]
    def _format(dt): return f"{dt.day} {nama_bulan[dt.month - 1]} {dt.year}, {dt.strftime('%H:%M')}"
    try:
        hasil = subprocess.run(["git", "log", "-1", "--format=%cI", "--", path_csv], capture_output=True, text=True, timeout=5)
        if hasil.stdout.strip(): return _format(_dt.fromisoformat(hasil.stdout.strip()))
    except Exception: pass
    try:
        return _format(_dt.fromtimestamp(os.path.getmtime(path_csv))) + " (perkiraan)"
    except Exception: return "tidak diketahui"

KOLOM_TEKS_CARI = ["NMDEPT", "NMSATKER", "PROVINSI", "KABKOTA", "FUNGSI", "SUBFUNGSI", "PROGRAM", "KEGIATAN", "OUTPUT", "AKUN"]

@st.cache_data(show_spinner="Menyiapkan data utama...")
def siapkan_data(df_mentah: pd.DataFrame) -> pd.DataFrame:
    if df_mentah.empty: return df_mentah
    d = df_mentah.copy()
    d["REALISASI"] = d[BULAN_KOLOM].sum(axis=1)
    d["SISA PAGU"] = d["PAGU"] - d["REALISASI"]
    if "JENIS BELANJA" in d.columns:
        d["LABEL_JENIS_BELANJA"] = d["JENIS BELANJA"].map(LABEL_JENIS_BELANJA_SINGKAT).fillna(d.get("LABEL_JENIS_BELANJA", d["JENIS BELANJA"]))
        
        # Kategori Level 1 untuk Pie Chart Interaktif
        jb_str = d["JENIS BELANJA"].astype(str)
        d["KELOMPOK_BELANJA"] = np.where(jb_str.str.startswith("5"), "Belanja KL", 
                                np.where(jb_str.str.startswith("6"), "Belanja TKD", "Lainnya"))

    # Mengambil Nama Pemda (Kolom K / Index 10) sesuai instruksi
    if len(df_mentah.columns) > 10:
        d["NAMA_PEMDA"] = df_mentah.iloc[:, 10].astype(str)
    else:
        d["NAMA_PEMDA"] = "Tidak Diketahui"

    kolom_ada = [c for c in KOLOM_TEKS_CARI if c in d.columns]
    d["_TEKS_CARI"] = d[kolom_ada].fillna("").astype(str).agg(" ".join, axis=1).str.lower() if kolom_ada else ""
    return d

# --------------------------------------------------------------------------
# Load data alternatif (Untuk Halaman 2 - Program Prioritas)
# --------------------------------------------------------------------------

@st.cache_data(show_spinner="Memuat data Program Prioritas...")
def load_data_prioritas() -> pd.DataFrame:
    """Memuat dari sumber berbeda. Fallback ke data dummy jika file tidak ada."""
    path = "data/program_prioritas.csv"
    try:
        df_pp = pd.read_csv(path)
        return df_pp
    except FileNotFoundError:
        # Data dummy agar dashboard tetap bisa didemonstrasikan
        return pd.DataFrame({
            "TAHUN": [2024, 2024, 2024, 2024, 2023, 2023],
            "NAMA_PROGRAM": [
                "Ketahanan Pangan Nasional", "Pembangunan Infrastruktur", 
                "Pendidikan Vokasi", "Kesehatan Masyarakat",
                "Ketahanan Pangan Nasional", "Pembangunan Infrastruktur"
            ],
            "KATEGORI": ["Pertanian", "Infrastruktur", "Pendidikan", "Kesehatan", "Pertanian", "Infrastruktur"],
            "PAGU": [5000000000, 15000000000, 3000000000, 8000000000, 4500000000, 12000000000],
            "REALISASI": [3500000000, 9000000000, 2800000000, 6000000000, 4400000000, 11500000000],
            "TARGET_OUTPUT": [100, 50, 200, 500, 90, 45],
            "REALISASI_OUTPUT": [75, 30, 190, 400, 90, 40]
        })

# Eksekusi penyiapan data utama
df = siapkan_data(load_data())

# --------------------------------------------------------------------------
# Login & Otentikasi
# --------------------------------------------------------------------------

SUPERUSER_USERNAME = "kanwil04"
SUPERUSER_PASSWORD = "admin"

def _cek_login(username: str, password: str, df_all: pd.DataFrame):
    username = (username or "").strip()
    password = (password or "").strip()
    if username == SUPERUSER_USERNAME and password == SUPERUSER_PASSWORD:
        return {"role": "super", "kdsatker": None}
    if username and username == password and username.isdigit():
        kdsatker = int(username)
        if "KDSATKER" in df_all.columns and kdsatker in df_all["KDSATKER"].unique():
            return {"role": "satker", "kdsatker": kdsatker}
    return None

if "auth" not in st.session_state:
    st.session_state.auth = None

if st.session_state.auth is None:
    st.title("🔐 Login Dashboard Terpadu")
    st.caption("Login memakai kode satker Anda sebagai username & password (atau gunakan superuser).")
    with st.form("form_login"):
        username_input = st.text_input("Username (kode satker, 6 digit)", placeholder="mis. 012345")
        password_input = st.text_input("Password", type="password")
        submit_login = st.form_submit_button("Login")
    if submit_login:
        hasil_login = _cek_login(username_input, password_input, df)
        if hasil_login:
            st.session_state.auth = hasil_login
            st.rerun()
        else:
            st.error("Username/password salah, atau kode satker tidak ditemukan.")
    st.stop()

auth = st.session_state.auth
is_super = auth["role"] == "super"
SCOPE_KDSATKER = None if is_super else auth["kdsatker"]

# --------------------------------------------------------------------------
# Komponen UI Reusable
# --------------------------------------------------------------------------

def kpi_card(label: str, value: str, delta: str = None, color="#0f172a"):
    delta_html = f'<div style="font-size:0.85rem;color:#16a34a;margin-top:4px;">{delta}</div>' if delta else ""
    st.markdown(
        f"""
        <div style="background:#f8fafc;border:1px solid #e2e8f0;border-radius:12px;
                    padding:16px 18px;min-height:110px;margin-bottom:1rem;">
            <div style="font-size:0.85rem;color:#64748b;margin-bottom:6px;">{label}</div>
            <div style="font-size:clamp(1rem, 2vw, 1.5rem);font-weight:700;color:{color};
                        white-space:normal;overflow-wrap:break-word;line-height:1.25;">{value}</div>
            {delta_html}
        </div>
        """,
        unsafe_allow_html=True,
    )

def get_groq_client():
    api_key = st.secrets.get("GROQ_API_KEY") if hasattr(st, "secrets") else None
    api_key = api_key or os.environ.get("GROQ_API_KEY")
    if not api_key: return None
    return Groq(api_key=api_key)

def render_ai_chat(page_id: str, context_data: str):
    st.divider()
    st.subheader("🤖 Dialog Asisten AI")
    st.caption("Tanyakan insight atau analisis terkait data pada halaman ini.")
    
    chat_key = f"chat_history_{page_id}"
    if chat_key not in st.session_state:
        st.session_state[chat_key] = []
        
    for msg in st.session_state[chat_key]:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            
    user_input = st.chat_input(f"Tanya AI untuk {page_id}...")
    if user_input:
        st.session_state[chat_key].append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)
            
        client = get_groq_client()
        if client:
            with st.chat_message("assistant"):
                with st.spinner("Menganalisis..."):
                    try:
                        sys_prompt = f"Anda adalah asisten cerdas penganalisis dashboard anggaran. Jawab dengan ringkas dan analitis. Konteks data: {context_data}"
                        messages = [{"role": "system", "content": sys_prompt}] + st.session_state[chat_key][-4:]
                        resp = client.chat.completions.create(model=GROQ_MODEL, messages=messages)
                        ans = resp.choices[0].message.content
                        st.markdown(ans)
                        st.session_state[chat_key].append({"role": "assistant", "content": ans})
                    except Exception as e:
                        st.error(f"Error memanggil AI: {e}")
        else:
            st.warning("API Key Groq tidak ditemukan di environment/secrets.")

# --------------------------------------------------------------------------
# HALAMAN 1: Dashboard Pagu & Realisasi Satker (Original)
# --------------------------------------------------------------------------

def page_dashboard_utama():
    st.title("📊 Dashboard Anggaran dan Transfer Riau Terkini")
    st.caption(f"🕒 Data terakhir diperbarui: {tanggal_update_data()}")

    if df.empty:
        st.warning("Data pagu_realisasi tidak tersedia atau kosong.")
        return

    st.sidebar.header("Filter Halaman Utama")
    
    # -- Logika Filter (dari kode asli) --
    if is_super:
        tahun_list = sorted(df["TAHUN"].unique(), reverse=True)
        tahun = st.sidebar.selectbox("Tahun", tahun_list, key="h1_tahun")
        df_tahun = df[df["TAHUN"] == tahun]

        SEMUA_DEPT = "— Semua Kementerian/Lembaga —"
        SEMUA_SATKER = "— Semua Satker —"

        dept_options = df_tahun[["KDDEPT", "NMDEPT"]].drop_duplicates().sort_values("KDDEPT")
        dept_options["LABEL"] = dept_options["KDDEPT"].apply(fmt_dept) + " - " + dept_options["NMDEPT"]
        dept_label = st.sidebar.selectbox("Kementerian/Lembaga", [SEMUA_DEPT] + dept_options["LABEL"].tolist(), key="h1_dept")

        if dept_label == SEMUA_DEPT:
            kddept, nmdept = None, "Semua Kementerian/Lembaga"
            df_dept = df_tahun
        else:
            kddept = int(dept_label.split(" - ")[0])
            nmdept = dept_options.loc[dept_options["KDDEPT"] == kddept, "NMDEPT"].iloc[0]
            df_dept = df_tahun[df_tahun["KDDEPT"] == kddept]

        satker_options = df_dept[["KDSATKER", "NMSATKER"]].drop_duplicates().sort_values("KDSATKER")
        satker_options["LABEL"] = satker_options["KDSATKER"].apply(fmt_satker) + " - " + satker_options["NMSATKER"]
        satker_label = st.sidebar.selectbox("Satuan Kerja (Satker)", [SEMUA_SATKER] + satker_options["LABEL"].tolist(), key="h1_satker")

        if satker_label == SEMUA_SATKER:
            kdsatker, nmsatker = None, "Semua Satker"
            df_satker = df_dept
        else:
            kdsatker = int(satker_label.split(" - ")[0])
            nmsatker = satker_options.loc[satker_options["KDSATKER"] == kdsatker, "NMSATKER"].iloc[0]
            df_satker = df_dept[df_dept["KDSATKER"] == kdsatker]
    else:
        kdsatker = auth["kdsatker"]
        df_kdsatker_semua_tahun = df[df["KDSATKER"] == kdsatker]
        tahun_list = sorted(df_kdsatker_semua_tahun["TAHUN"].unique(), reverse=True)
        if not tahun_list:
            st.error(f"Tidak ada data untuk satker dengan kode {fmt_satker(kdsatker)}.")
            return
        tahun = st.sidebar.selectbox("Tahun", tahun_list, key="h1_tahun")
        df_tahun = df[df["TAHUN"] == tahun]
        df_satker = df_tahun[df_tahun["KDSATKER"] == kdsatker]

        if df_satker.empty:
            st.warning(f"Satker Anda belum punya data di tahun {tahun}.")
            return
        nmsatker = df_satker["NMSATKER"].iloc[0]
        kddept = int(df_satker["KDDEPT"].iloc[0])
        nmdept = df_satker["NMDEPT"].iloc[0]

    st.caption(f"**{nmdept}** — **{nmsatker}** — Tahun {tahun}")

    # Inisialisasi state filter pie chart interaktif
    if "filter_kategori_belanja" not in st.session_state:
        st.session_state.filter_kategori_belanja = "Semua"
        
    filter_aktif = st.session_state.filter_kategori_belanja
    
    # Terapkan filter berdasarkan hasil klik Pie Chart ke dataset
    if filter_aktif != "Semua":
        df_satker_filtered = df_satker[df_satker["KELOMPOK_BELANJA"] == filter_aktif]
    else:
        df_satker_filtered = df_satker.copy()

    # -- Agregasi (menggunakan data yang sudah disaring via session state) --
    pagu_total = df_satker_filtered["PAGU"].sum()
    realisasi_total = df_satker_filtered["REALISASI"].sum()
    sisa_pagu = df_satker_filtered["SISA PAGU"].sum()
    persen_serapan = (realisasi_total / pagu_total * 100) if pagu_total else 0
    monthly = df_satker_filtered[BULAN_KOLOM].sum()
    kumulatif = monthly.cumsum()

    bulan_terisi = [i + 1 for i, v in enumerate(monthly.values) if v != 0]
    bulan_terakhir = max(bulan_terisi) if bulan_terisi else 0

    hari_ini = date.today()
    if tahun < hari_ini.year: bulan_penuh_terakhir = 12
    elif tahun > hari_ini.year: bulan_penuh_terakhir = 0
    else: bulan_penuh_terakhir = min(bulan_terakhir, hari_ini.month - 1)

    jenis_belanja = df_satker.groupby("LABEL_JENIS_BELANJA")["REALISASI"].sum().sort_values(ascending=False).reset_index()

    # -- Proyeksi Logik (Disesuaikan ke local scope) --
    BOBOT_TAHUN = {1: 0.50, 2: 0.25, 3: 0.125, 4: 0.0625, 5: 0.0625}
    def _filter_entitas_lokal(thn: int) -> pd.DataFrame:
        d = df[df["TAHUN"] == thn]
        if kddept is not None: d = d[d["KDDEPT"] == kddept]
        if kdsatker is not None: d = d[d["KDSATKER"] == kdsatker]
        return d

    def hitung_proyeksi_agregat_lokal(tahun_y: int, pagu_y: float):
        total_rate, total_bobot, tahun_dipakai = np.zeros(12), 0.0, []
        for i in range(1, 6):
            d_prev = _filter_entitas_lokal(tahun_y - i)
            pagu_prev = d_prev["PAGU"].sum() if not d_prev.empty else 0
            if pagu_prev <= 0: continue
            total_rate += BOBOT_TAHUN[i] * (d_prev[BULAN_KOLOM].sum().values.astype(float) / pagu_prev)
            total_bobot += BOBOT_TAHUN[i]
            tahun_dipakai.append(tahun_y - i)
        if total_bobot == 0: return None, tahun_dipakai
        return (total_rate / total_bobot) * pagu_y, tahun_dipakai

    proyeksi_agregat_bulanan, tahun_dipakai = hitung_proyeksi_agregat_lokal(tahun, pagu_total)
    
    if proyeksi_agregat_bulanan is None:
        rerata_bulanan = (kumulatif.iloc[bulan_terakhir - 1] / bulan_terakhir) if bulan_terakhir else 0
        proyeksi_akhir_tahun = rerata_bulanan * 12
        metode_proyeksi = "fallback"
    else:
        target_tahun_penuh = proyeksi_agregat_bulanan.sum()
        proyeksi_akhir_tahun = max(realisasi_total, target_tahun_penuh)
        metode_proyeksi = "historis"
    persen_proyeksi = (proyeksi_akhir_tahun / pagu_total * 100) if pagu_total else 0

    # -- Visualisasi Halaman 1 --
    r1c1, r1c2, r2c1, r2c2 = st.columns(4)
    with r1c1: kpi_card("Total Pagu", f"Rp {pagu_total:,.0f}")
    with r1c2: kpi_card("Realisasi", f"Rp {realisasi_total:,.0f}", f"{persen_serapan:.1f}%")
    with r2c1: kpi_card("Sisa Pagu", f"Rp {sisa_pagu:,.0f}")
    with r2c2: kpi_card("Proyeksi Akhir", f"Rp {proyeksi_akhir_tahun:,.0f}", f"{persen_proyeksi:.1f}%")

    st.divider()
    col1, col2 = st.columns([3, 2])
    with col1:
        st.subheader("Realisasi per Bulan")
        fig_bar = px.bar(pd.DataFrame({"Bulan": BULAN_KOLOM, "Realisasi": monthly.values}), x="Bulan", y="Realisasi", text_auto=".2s")
        st.plotly_chart(fig_bar, use_container_width=True)
    with col2:
        st.subheader("Komposisi Sisa")
        fig_pie1 = px.pie(names=["Realisasi", "Sisa Pagu"], values=[realisasi_total, max(sisa_pagu, 0)], hole=0.4)
        st.plotly_chart(fig_pie1, use_container_width=True)

    st.subheader(f"Komposisi Belanja {'' if filter_aktif == 'Semua' else f'- {filter_aktif}'}")
    st.caption("💡 Klik pada bagian chart untuk melihat rincian detail (drilldown), serta menyaring grafik & KPI di halaman ini.")
    
    if filter_aktif != "Semua":
        if st.button("⬅️ Kembali Tampilkan Semua Belanja"):
            st.session_state.filter_kategori_belanja = "Semua"
            st.rerun()

    # Tentukan Level Data Pie Chart
    if filter_aktif == "Semua":
        # Level 1: Belanja KL vs Belanja TKD
        dist_belanja = df_satker.groupby("KELOMPOK_BELANJA")["REALISASI"].sum().reset_index()
        
        # Filter hanya tampilkan Belanja KL dan Belanja TKD (sesuai instruksi)
        dist_belanja = dist_belanja[dist_belanja["KELOMPOK_BELANJA"].isin(["Belanja KL", "Belanja TKD"])]
        
        fig_pie2 = px.pie(dist_belanja, names="KELOMPOK_BELANJA", values="REALISASI", hole=0.4, 
                          color="KELOMPOK_BELANJA", color_discrete_map={"Belanja KL": "#3b82f6", "Belanja TKD": "#10b981"})
    else:
        # Level 2: Detail per Jenis Belanja (Pegawai, Barang, Modal, Bansos / DBH, DAU, dll)
        dist_belanja = df_satker_filtered.groupby("LABEL_JENIS_BELANJA")["REALISASI"].sum().reset_index()
        fig_pie2 = px.pie(dist_belanja, names="LABEL_JENIS_BELANJA", values="REALISASI", hole=0.4)

    try:
        # Menggunakan fitur on_select interaktif Streamlit (v1.35+)
        event = st.plotly_chart(fig_pie2, use_container_width=True, on_select="rerun", key="pie_chart_h1")
        
        pts = []
        if isinstance(event, dict):
            pts = event.get("selection", {}).get("points", [])
        elif hasattr(event, "selection"):
            pts = getattr(event, "selection", {}).get("points", [])
            
        if pts:
            clicked_label = pts[0].get("label", pts[0].get("x"))
            
            # Jika di Level 1 dan mengklik KL/TKD, terapkan filter state dan rerun aplikasinya
            if filter_aktif == "Semua" and clicked_label in ["Belanja KL", "Belanja TKD"]:
                st.session_state.filter_kategori_belanja = clicked_label
                st.rerun()
    except Exception:
        # Fallback manual jika versi Streamlit pengguna masih lama
        st.plotly_chart(fig_pie2, use_container_width=True)
        st.info("Pilih kategori di bawah untuk memfilter data:")
        fallback_choice = st.radio("Kategori Belanja:", ["Semua", "Belanja KL", "Belanja TKD"], 
                                   index=["Semua", "Belanja KL", "Belanja TKD"].index(filter_aktif), horizontal=True)
        if fallback_choice != filter_aktif:
            st.session_state.filter_kategori_belanja = fallback_choice
            st.rerun()

    # -- AI Chat Area (Menggunakan komponen reusable) --
    context_h1 = f"Satker: {nmsatker} ({tahun}). Pagu (berdasar filter {filter_aktif}): {pagu_total:,.0f}. Realisasi: {realisasi_total:,.0f} ({persen_serapan:.1f}%)."
    render_ai_chat("halaman1_utama", context_h1)


# --------------------------------------------------------------------------
# HALAMAN 2: Dashboard Program Prioritas
# --------------------------------------------------------------------------

def page_program_prioritas():
    st.title("🎯 Dashboard Program Prioritas")
    st.caption("Sumber Data: `program_prioritas.csv` (Terpisah dari data satker utama)")
    
    df_pp = load_data_prioritas()
    
    if df_pp.empty:
        st.warning("Data program prioritas kosong.")
        return

    st.sidebar.header("Filter Program")
    tahun_list = sorted(df_pp["TAHUN"].unique(), reverse=True)
    tahun = st.sidebar.selectbox("Tahun", tahun_list, key="h2_tahun")
    
    df_filtered = df_pp[df_pp["TAHUN"] == tahun]
    
    kategori_list = ["Semua Kategori"] + list(df_filtered["KATEGORI"].unique())
    kategori = st.sidebar.selectbox("Kategori", kategori_list, key="h2_kat")
    if kategori != "Semua Kategori":
        df_filtered = df_filtered[df_filtered["KATEGORI"] == kategori]

    # Menghitung Metrik
    total_pagu = df_filtered["PAGU"].sum()
    total_realisasi = df_filtered["REALISASI"].sum()
    persentase = (total_realisasi / total_pagu * 100) if total_pagu > 0 else 0
    
    # Menampilkan KPI
    c1, c2, c3 = st.columns(3)
    with c1: kpi_card("Total Alokasi Program", f"Rp {total_pagu:,.0f}", color="#1e3a8a")
    with c2: kpi_card("Total Terserap", f"Rp {total_realisasi:,.0f}", f"{persentase:.1f}%", color="#047857")
    with c3: kpi_card("Jumlah Program", str(len(df_filtered)), color="#b45309")
    
    st.divider()
    
    # Chart
    st.subheader(f"Capaian per Program Prioritas ({tahun})")
    
    if not df_filtered.empty:
        # Sort by Pagu
        df_plot = df_filtered.sort_values(by="PAGU", ascending=True)
        
        fig = go.Figure()
        fig.add_trace(go.Bar(
            y=df_plot["NAMA_PROGRAM"], x=df_plot["PAGU"], 
            name="Alokasi (Pagu)", orientation='h', marker_color="#94a3b8"
        ))
        fig.add_trace(go.Bar(
            y=df_plot["NAMA_PROGRAM"], x=df_plot["REALISASI"], 
            name="Realisasi", orientation='h', marker_color="#2563eb"
        ))
        
        fig.update_layout(barmode='group', height=400 + (len(df_plot) * 20))
        st.plotly_chart(fig, use_container_width=True)
        
        # Tabel Data
        st.markdown("**Detail Program**")
        tabel_tampil = df_filtered[["NAMA_PROGRAM", "KATEGORI", "PAGU", "REALISASI", "TARGET_OUTPUT", "REALISASI_OUTPUT"]].copy()
        tabel_tampil["% Serapan"] = (tabel_tampil["REALISASI"] / tabel_tampil["PAGU"] * 100).round(2)
        st.dataframe(tabel_tampil, use_container_width=True, hide_index=True)

    # Tambahan Chat Box AI untuk Halaman 2
    context_h2 = f"Total Pagu Program: {total_pagu}. Realisasi: {total_realisasi} ({persentase:.1f}%). Jumlah Program Prioritas: {len(df_filtered)}."
    render_ai_chat("halaman2_prioritas", context_h2)


# --------------------------------------------------------------------------
# HALAMAN 3: Dashboard Dana Transfer ke Daerah (TKD)
# --------------------------------------------------------------------------

def page_tkd():
    st.title("💸 Dana Transfer ke Daerah (TKD)")
    st.caption("Sumber Data: Terintegrasi dari dataset satker utama (Jenis Belanja 61-66)")
    
    if df.empty:
        st.warning("Data belum tersedia.")
        return
        
    # Daftar Jenis Belanja yang termasuk TKD (diperbarui)
    tkd_labels = ["DBH", "DAU", "DAK Fisik", "Insentif Daerah", "DAK Nonfisik", "Dana Desa"]
    
    # Filter global df untuk TKD
    if "LABEL_JENIS_BELANJA" in df.columns:
        df_tkd = df[df["LABEL_JENIS_BELANJA"].isin(tkd_labels)]
    else:
        df_tkd = df[df["JENIS BELANJA"].astype(str).str.startswith("6", na=False)] # Alternatif jika label belum ter-map

    if df_tkd.empty:
        st.info("Tidak ada transaksi Dana Transfer ke Daerah (TKD) yang ditemukan pada dataset ini.")
        return
        
    st.sidebar.header("Filter TKD")
    
    # Pembatasan scope (khusus satker vs superuser)
    if is_super:
        df_scope = df_tkd
    else:
        df_scope = df_tkd[df_tkd["KDSATKER"] == auth["kdsatker"]]
        
    if df_scope.empty:
        st.warning("Satker Anda tidak memiliki alokasi TKD.")
        return

    tahun_list = sorted(df_scope["TAHUN"].unique(), reverse=True)
    tahun = st.sidebar.selectbox("Tahun Anggaran", tahun_list, key="h3_tahun")
    
    df_filtered = df_scope[df_scope["TAHUN"] == tahun]
    
    # Aggregate
    total_pagu_tkd = df_filtered["PAGU"].sum()
    total_real_tkd = df_filtered["REALISASI"].sum()
    persen_tkd = (total_real_tkd / total_pagu_tkd * 100) if total_pagu_tkd > 0 else 0
    
    col1, col2 = st.columns(2)
    with col1: kpi_card("Total Alokasi TKD", f"Rp {total_pagu_tkd:,.0f}", color="#4338ca")
    with col2: kpi_card("Total Disalurkan", f"Rp {total_real_tkd:,.0f}", f"{persen_tkd:.1f}%", color="#059669")
    
    st.divider()
    
    row1, row2 = st.columns([1, 1])
    
    with row1:
        st.subheader("Distribusi Jenis TKD")
        dist_tkd = df_filtered.groupby("LABEL_JENIS_BELANJA")["PAGU"].sum().reset_index()
        if not dist_tkd.empty:
            fig_pie = px.pie(dist_tkd, names="LABEL_JENIS_BELANJA", values="PAGU", hole=0.3)
            st.plotly_chart(fig_pie, use_container_width=True)
            
    with row2:
        st.subheader("Penyaluran per Bulan")
        monthly_tkd = df_filtered[BULAN_KOLOM].sum()
        df_monthly = pd.DataFrame({"Bulan": BULAN_KOLOM, "Penyaluran": monthly_tkd.values})
        fig_line = px.line(df_monthly, x="Bulan", y="Penyaluran", markers=True)
        fig_line.update_traces(line_color="#059669", line_width=3)
        st.plotly_chart(fig_line, use_container_width=True)

    st.divider()
    st.subheader("Pembagian TKD Berdasarkan Nama Pemda")
    if "NAMA_PEMDA" in df_filtered.columns:
        df_pemda = df_filtered.groupby("NAMA_PEMDA").agg(
            Alokasi_TKD=("PAGU", "sum"),
            Realisasi_TKD=("REALISASI", "sum")
        ).reset_index()
        df_pemda["% Serapan"] = (df_pemda["Realisasi_TKD"] / df_pemda["Alokasi_TKD"] * 100).fillna(0).round(2)
        df_pemda = df_pemda.sort_values(by="Alokasi_TKD", ascending=False)
        
        st.dataframe(df_pemda.style.format({
            "Alokasi_TKD": "Rp {:,.0f}",
            "Realisasi_TKD": "Rp {:,.0f}",
            "% Serapan": "{:.2f}%"
        }), use_container_width=True, hide_index=True)
    else:
        st.info("Informasi Nama Pemda tidak tersedia pada dataset.")

    # Tambahan Chat Box AI untuk Halaman 3
    context_h3 = f"Total Alokasi TKD: {total_pagu_tkd}. Tersalurkan: {total_real_tkd} ({persen_tkd:.1f}%)."
    render_ai_chat("halaman3_tkd", context_h3)


# --------------------------------------------------------------------------
# Main Router (Navigasi Halaman)
# --------------------------------------------------------------------------

with st.sidebar:
    st.markdown("### Profil Pengguna")
    if is_super:
        st.success(f"👤 Super User ({SUPERUSER_USERNAME})")
    else:
        st.success(f"👤 Satker: {fmt_satker(auth['kdsatker'])}")
    if st.button("🚪 Logout", use_container_width=True):
        st.session_state.auth = None
        st.rerun()
        
    st.markdown("---")
    st.markdown("### Menu Navigasi")
    # Pilihan Halaman
    halaman = st.radio(
        "Pilih Halaman:",
        [
            "1. Pagu & Realisasi Satker", 
            "2. Program Prioritas", 
            "3. Dana Transfer ke Daerah"
        ],
        label_visibility="collapsed"
    )
    st.markdown("---")

# Merutekan ke fungsi yang sesuai berdasarkan pilihan
if halaman == "1. Pagu & Realisasi Satker":
    page_dashboard_utama()
elif halaman == "2. Program Prioritas":
    page_program_prioritas()
elif halaman == "3. Dana Transfer ke Daerah":
    page_tkd()