"""
Dashboard Pagu & Realisasi Satker (Multi-Page dengan AI)
--------------------------------------------------------
Sistem dashboard 4 halaman dengan integrasi Groq AI di setiap halamannya.
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
    page_title="Dashboard Pagu & Realisasi",
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
    61: "TKD DBH",
    62: "TKD DAU",
    63: "TKD DAK Fisik",
    64: "TKD Insentif Fiskal",
    65: "TKD DAK Nonfisik",
    66: "TKD Dana Desa",
}

GROQ_MODEL = os.environ.get("GROQ_MODEL", "llama3-70b-8192") # Opsional ganti model

def fmt_satker(kode) -> str:
    if kode is None: return ""
    try: return f"{int(kode):06d}"
    except (TypeError, ValueError): return str(kode)

def fmt_dept(kode) -> str:
    if kode is None: return ""
    try: return f"{int(kode):03d}"
    except (TypeError, ValueError): return str(kode)

# --------------------------------------------------------------------------
# Load data (CSV, Supabase, dan Excel)
# --------------------------------------------------------------------------
@st.cache_data(show_spinner="Memuat data utama...")
def load_data_from_csv(path: str) -> pd.DataFrame:
    try:
        return pd.read_csv(path)
    except Exception as e:
        st.error(f"Gagal memuat file utama: {e}")
        return pd.DataFrame()

@st.cache_data(show_spinner="Memuat data excel...")
def load_data_from_excel(path: str) -> pd.DataFrame:
    try:
        return pd.read_excel(path)
    except FileNotFoundError:
        st.warning(f"File '{path}' tidak ditemukan. Mohon pastikan file tersedia di direktori yang sama.")
        return pd.DataFrame()
    except Exception as e:
        st.error(f"Gagal membaca excel {path}: {e}")
        return pd.DataFrame()

def load_data() -> pd.DataFrame:
    # Hapus logika supabase sementara agar fokus pada CSV/Excel sesuai permintaan
    return load_data_from_csv("data/pagu_realisasi.csv.gz")

KOLOM_TEKS_CARI = [
    "NMDEPT", "NMSATKER", "PROVINSI", "KABKOTA", "FUNGSI", "SUBFUNGSI",
    "PROGRAM", "KEGIATAN", "OUTPUT", "AKUN", "kabkota_uraian"
]

@st.cache_data(show_spinner="Menyiapkan data utama...")
def siapkan_data(df_mentah: pd.DataFrame) -> pd.DataFrame:
    if df_mentah.empty: return df_mentah
    d = df_mentah.copy()
    
    # Pastikan kolom bulan ada, jika tidak isi 0
    for b in BULAN_KOLOM:
        if b not in d.columns: d[b] = 0
        
    d["REALISASI"] = d[BULAN_KOLOM].sum(axis=1)
    d["SISA PAGU"] = d["PAGU"] - d["REALISASI"]

    if "JENIS BELANJA" in d.columns:
        d["LABEL_JENIS_BELANJA"] = d["JENIS BELANJA"].map(LABEL_JENIS_BELANJA_SINGKAT).fillna(d.get("LABEL_JENIS_BELANJA", "Lainnya"))
        # Klasifikasi KL vs TKD berdasarkan awalan angka
        d["KELOMPOK_BELANJA"] = np.where(
            d["JENIS BELANJA"].astype(str).str.startswith("5"), "Belanja KL",
            np.where(d["JENIS BELANJA"].astype(str).str.startswith("6"), "Belanja TKD", "Lainnya")
        )
    else:
        d["KELOMPOK_BELANJA"] = "Lainnya"

    kolom_ada = [c for c in KOLOM_TEKS_CARI if c in d.columns]
    if kolom_ada:
        d["_TEKS_CARI"] = d[kolom_ada].fillna("").astype(str).agg(" ".join, axis=1).str.lower()
    else:
        d["_TEKS_CARI"] = ""
    return d

df = siapkan_data(load_data())
df_prioritas = load_data_from_excel("prioritas presiden 2026.xlsx")
df_strategis = load_data_from_excel("program strategis2026.xlsx")

# --------------------------------------------------------------------------
# Login & State
# --------------------------------------------------------------------------
SUPERUSER_USERNAME = "kanwil04"
SUPERUSER_PASSWORD = "admin"

def _cek_login(username, password, df_all):
    username = (username or "").strip()
    password = (password or "").strip()
    if username == SUPERUSER_USERNAME and password == SUPERUSER_PASSWORD:
        return {"role": "super", "kdsatker": None}
    if username and username == password and username.isdigit():
        kdsatker = int(username)
        if not df_all.empty and kdsatker in df_all["KDSATKER"].unique():
            return {"role": "satker", "kdsatker": kdsatker}
    return None

if "auth" not in st.session_state:
    st.session_state.auth = None

if st.session_state.auth is None:
    st.title("🔐 Login Dashboard Pagu & Realisasi")
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

# --------------------------------------------------------------------------
# Navigasi & Sidebar
# --------------------------------------------------------------------------
with st.sidebar:
    if is_super:
        st.success(f"👤 Super User ({SUPERUSER_USERNAME})")
    else:
        st.success(f"👤 Satker: {fmt_satker(auth['kdsatker'])}")
    
    st.markdown("### 🧭 Navigasi Menu")
    menu = st.radio(
        "Pilih Halaman:",
        ["1. Pagu & Realisasi", "2. Prioritas Presiden", "3. Program Strategis", "4. Transfer ke Daerah"],
        label_visibility="collapsed"
    )
    st.divider()

    if st.button("🚪 Logout"):
        st.session_state.auth = None
        st.rerun()

# --------------------------------------------------------------------------
# Komponen Global: KPI Card & Chat AI
# --------------------------------------------------------------------------
def kpi_card(label: str, value: str, delta: str = None):
    delta_html = f'<div style="font-size:0.85rem;color:#16a34a;margin-top:4px;">{delta}</div>' if delta else ""
    st.markdown(
        f"""
        <div style="background:#f8fafc;border:1px solid #e2e8f0;border-radius:12px;
                    padding:16px 18px;min-height:110px;margin-bottom:1rem;">
            <div style="font-size:0.85rem;color:#64748b;margin-bottom:6px;">{label}</div>
            <div style="font-size:clamp(1rem, 2.1vw, 1.6rem);font-weight:700;color:#0f172a;
                        white-space:normal;overflow-wrap:break-word;line-height:1.25;">{value}</div>
            {delta_html}
        </div>
        """,
        unsafe_allow_html=True,
    )

def get_groq_client():
    api_key = st.secrets.get("GROQ_API_KEY") if hasattr(st, "secrets") else os.environ.get("GROQ_API_KEY")
    if not api_key: return None
    return Groq(api_key=api_key)

def render_chat_ai(context_info: str, page_name: str):
    """Merender chatbox AI di bagian bawah halaman manapun."""
    st.divider()
    st.subheader(f"🤖 Tanya AI tentang {page_name}")
    
    chat_key = f"chat_history_{page_name}"
    if chat_key not in st.session_state:
        st.session_state[chat_key] = []

    col_title, col_reset = st.columns([5, 1])
    with col_reset:
        if st.button("🗑️ Reset Chat", key=f"reset_{page_name}"):
            st.session_state[chat_key] = []
            st.rerun()

    for msg in st.session_state[chat_key]:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])

    prompt = st.chat_input(f"Tanyakan sesuatu terkait data di halaman {page_name}...")
    
    if prompt:
        st.session_state[chat_key].append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.write(prompt)

        client = get_groq_client()
        if client is None:
            jawaban = "⚠️ API Key Groq belum diatur. Setel `GROQ_API_KEY`."
        else:
            with st.spinner("AI sedang berpikir..."):
                system_msg = {
                    "role": "system",
                    "content": (
                        "Kamu adalah asisten analis anggaran pemerintah yang cerdas. "
                        f"User saat ini sedang melihat halaman: '{page_name}'. "
                        "Gunakan data ringkasan berikut untuk menjawab pertanyaan user dengan jelas dan ringkas. "
                        f"\n\n--- KONTEKS DATA ---\n{context_info}"
                    )
                }
                messages = [system_msg] + st.session_state[chat_key][-6:]
                
                try:
                    resp = client.chat.completions.create(model=GROQ_MODEL, messages=messages)
                    jawaban = resp.choices[0].message.content
                except Exception as e:
                    jawaban = f"⚠️ Gagal menghubungi AI: {str(e)}"

        st.session_state[chat_key].append({"role": "assistant", "content": jawaban})
        with st.chat_message("assistant"):
            st.write(jawaban)


# --------------------------------------------------------------------------
# HALAMAN 1: Dashboard Pagu & Realisasi
# --------------------------------------------------------------------------
def render_page_1():
    st.title("📊 Dashboard Pagu & Realisasi Satker")
    
    if df.empty:
        st.warning("Data pagu_realisasi.csv.gz kosong atau gagal dimuat.")
        return

    # Filter Global Halaman 1
    col_f1, col_f2 = st.columns(2)
    tahun_list = sorted(df["TAHUN"].unique(), reverse=True)
    with col_f1:
        tahun = st.selectbox("Pilih Tahun Anggaran", tahun_list)
    
    df_tahun = df[df["TAHUN"] == tahun]
    
    if is_super:
        with col_f2:
            dept_opts = ["— Semua Kementerian/Lembaga —"] + list(df_tahun["NMDEPT"].dropna().unique())
            dept = st.selectbox("Kementerian / Lembaga", dept_opts)
        
        if dept != "— Semua Kementerian/Lembaga —":
            df_tahun = df_tahun[df_tahun["NMDEPT"] == dept]
    else:
        df_tahun = df_tahun[df_tahun["KDSATKER"] == auth["kdsatker"]]

    # --- FITUR BARU: DRILL DOWN KOMPOSISI BELANJA (Mempengaruhi semua grafik & KPI) ---
    st.markdown("### 🎯 Fokus Analisis Belanja")
    analisis_mode = st.radio(
        "Filter Data Berdasarkan Kelompok Belanja:",
        ["Gabungan (KL & TKD)", "Hanya Belanja KL (Awalan Akun 5)", "Hanya Belanja TKD (Awalan Akun 6)"],
        horizontal=True
    )

    df_filtered = df_tahun.copy()
    if analisis_mode == "Hanya Belanja KL (Awalan Akun 5)":
        df_filtered = df_filtered[df_filtered["KELOMPOK_BELANJA"] == "Belanja KL"]
    elif analisis_mode == "Hanya Belanja TKD (Awalan Akun 6)":
        df_filtered = df_filtered[df_filtered["KELOMPOK_BELANJA"] == "Belanja TKD"]

    # Kalkulasi Agregat
    pagu_total = df_filtered["PAGU"].sum()
    realisasi_total = df_filtered["REALISASI"].sum()
    sisa_pagu = df_filtered["SISA PAGU"].sum()
    persen_serapan = (realisasi_total / pagu_total * 100) if pagu_total else 0
    monthly = df_filtered[BULAN_KOLOM].sum()

    # Tampilkan KPI
    r1c1, r1c2, r1c3, r1c4 = st.columns(4)
    with r1c1: kpi_card("Pagu Anggaran", f"Rp {pagu_total:,.0f}")
    with r1c2: kpi_card("Realisasi", f"Rp {realisasi_total:,.0f}", f"{persen_serapan:.1f}%")
    with r1c3: kpi_card("Sisa Pagu", f"Rp {sisa_pagu:,.0f}")
    with r1c4: kpi_card("Jumlah Baris Data", f"{len(df_filtered)} baris")

    # Grafik Utama
    col_g1, col_g2 = st.columns([3, 2])
    with col_g1:
        st.subheader("Tren Realisasi per Bulan")
        bar_df = pd.DataFrame({"Bulan": BULAN_KOLOM, "Realisasi": monthly.values})
        fig_bar = px.bar(bar_df, x="Bulan", y="Realisasi", text_auto=".2s")
        st.plotly_chart(fig_bar, use_container_width=True)

    with col_g2:
        st.subheader("Komposisi Belanja")
        # Logika dinamis untuk Pie Chart
        if analisis_mode == "Gabungan (KL & TKD)":
            # Tampilkan 2 saja: KL dan TKD
            pie_data = df_filtered.groupby("KELOMPOK_BELANJA")["REALISASI"].sum().reset_index()
            fig_pie = px.pie(pie_data, names="KELOMPOK_BELANJA", values="REALISASI", hole=0.4)
        else:
            # Tampilkan rincian (Pegawai, Barang, Modal, DBH, DAU, dll)
            pie_data = df_filtered.groupby("LABEL_JENIS_BELANJA")["REALISASI"].sum().reset_index()
            fig_pie = px.pie(pie_data, names="LABEL_JENIS_BELANJA", values="REALISASI", hole=0.4)
            
        st.plotly_chart(fig_pie, use_container_width=True)
        st.caption("*Mengubah filter 'Fokus Analisis' di atas akan merubah rincian chart ini.*")

    # Konteks untuk AI Halaman 1
    ai_context = f"""
    Tahun: {tahun}
    Mode Analisis: {analisis_mode}
    Total Pagu: Rp {pagu_total:,.0f}
    Total Realisasi: Rp {realisasi_total:,.0f} ({persen_serapan:.1f}%)
    Sisa Pagu: Rp {sisa_pagu:,.0f}
    """
    render_chat_ai(ai_context, "Halaman Pagu & Realisasi")

# --------------------------------------------------------------------------
# HALAMAN 2: Prioritas Presiden
# --------------------------------------------------------------------------
def render_page_2():
    st.title("🇮🇩 Dashboard Prioritas Presiden 2026")
    st.markdown("Sumber data: `prioritas presiden 2026.xlsx`")
    
    if df_prioritas.empty:
        st.warning("Data tidak tersedia atau file belum diunggah.")
        render_chat_ai("Data prioritas presiden kosong/tidak ditemukan.", "Prioritas Presiden")
        return

    st.dataframe(df_prioritas, use_container_width=True)
    
    # Deteksi otomatis kolom numerik untuk chart sederhana
    num_cols = df_prioritas.select_dtypes(include=[np.number]).columns
    cat_cols = df_prioritas.select_dtypes(exclude=[np.number]).columns
    
    if len(num_cols) > 0 and len(cat_cols) > 0:
        st.subheader("Ringkasan Visual")
        x_col = cat_cols[0]
        y_col = num_cols[0]
        agg_df = df_prioritas.groupby(x_col)[y_col].sum().reset_index().sort_values(y_col, ascending=False).head(10)
        fig = px.bar(agg_df, x=x_col, y=y_col, title=f"Top 10 {y_col} berdasarkan {x_col}")
        st.plotly_chart(fig, use_container_width=True)

    # Buat summary teks (maks 5 baris pertama) untuk dibaca AI
    sample_data = df_prioritas.head(5).to_csv(index=False)
    ai_context = f"Data Prioritas Presiden 2026. Terdapat {len(df_prioritas)} program/baris.\nSample Data:\n{sample_data}"
    render_chat_ai(ai_context, "Prioritas Presiden")


# --------------------------------------------------------------------------
# HALAMAN 3: Program Strategis
# --------------------------------------------------------------------------
def render_page_3():
    st.title("🎯 Dashboard Program Strategis 2026")
    st.markdown("Sumber data: `program strategis2026.xlsx`")

    if df_strategis.empty:
        st.warning("Data tidak tersedia atau file belum diunggah.")
        render_chat_ai("Data program strategis kosong/tidak ditemukan.", "Program Strategis")
        return

    st.dataframe(df_strategis, use_container_width=True)

    num_cols = df_strategis.select_dtypes(include=[np.number]).columns
    cat_cols = df_strategis.select_dtypes(exclude=[np.number]).columns
    
    if len(num_cols) > 0 and len(cat_cols) > 0:
        st.subheader("Distribusi Anggaran/Target")
        fig2 = px.pie(df_strategis.head(15), names=cat_cols[0], values=num_cols[0], hole=0.3, title=f"Komposisi {num_cols[0]}")
        st.plotly_chart(fig2, use_container_width=True)

    sample_data = df_strategis.head(5).to_csv(index=False)
    ai_context = f"Data Program Strategis 2026. Terdapat {len(df_strategis)} baris data.\nSample Data:\n{sample_data}"
    render_chat_ai(ai_context, "Program Strategis")


# --------------------------------------------------------------------------
# HALAMAN 4: Dana Transfer ke Daerah (TKD)
# --------------------------------------------------------------------------
def render_page_4():
    st.title("🏙️ Dashboard Dana Transfer ke Daerah (TKD)")
    st.markdown("Menampilkan rincian khusus belanja TKD (Akun awalan 6) berdasarkan Pemerintah Daerah.")

    if df.empty:
        st.warning("Data utama kosong.")
        return

    # Filter khusus TKD
    df_tkd = df[df["KELOMPOK_BELANJA"] == "Belanja TKD"]
    
    if df_tkd.empty:
        st.info("Tidak ada data belanja TKD (Awalan akun 6) yang ditemukan.")
        return

    # Hitung total
    total_pagu_tkd = df_tkd["PAGU"].sum()
    total_real_tkd = df_tkd["REALISASI"].sum()

    col1, col2 = st.columns(2)
    with col1: kpi_card("Total Pagu TKD", f"Rp {total_pagu_tkd:,.0f}")
    with col2: kpi_card("Total Realisasi TKD", f"Rp {total_real_tkd:,.0f}", f"{(total_real_tkd/total_pagu_tkd*100) if total_pagu_tkd else 0:.1f}%")

    st.subheader("Distribusi TKD per Pemerintah Daerah")
    
    # Cek apakah kolom kabkota_uraian ada
    kolom_pemda = "kabkota_uraian" if "kabkota_uraian" in df_tkd.columns else ("KABKOTA" if "KABKOTA" in df_tkd.columns else None)

    if kolom_pemda:
        # Agregasi data per Pemda
        df_pemda = df_tkd.groupby(kolom_pemda).agg(
            Pagu=("PAGU", "sum"),
            Realisasi=("REALISASI", "sum")
        ).reset_index()
        
        df_pemda["% Serapan"] = (df_pemda["Realisasi"] / df_pemda["Pagu"] * 100).round(2)
        df_pemda = df_pemda.sort_values("Pagu", ascending=False)
        
        st.dataframe(
            df_pemda.style.format({"Pagu": "Rp {:,.0f}", "Realisasi": "Rp {:,.0f}", "% Serapan": "{:.2f} %"}),
            use_container_width=True, height=400
        )

        # Bar chart Pemda tertinggi
        fig_pemda = px.bar(df_pemda.head(15), x=kolom_pemda, y="Pagu", title="Top 15 Pemda dengan Pagu TKD Terbesar", text_auto=".2s")
        st.plotly_chart(fig_pemda, use_container_width=True)

        # Context AI
        top3 = df_pemda.head(3).to_dict(orient="records")
        ai_context = f"Total Pagu TKD: {total_pagu_tkd}, Realisasi: {total_real_tkd}. Top 3 Pemda: {top3}"
    else:
        st.warning("Kolom 'kabkota_uraian' atau 'KABKOTA' tidak ditemukan di dataset untuk pemetaan Pemda.")
        ai_context = "Data TKD tersedia namun tidak ada kolom identifikasi Pemda."

    render_chat_ai(ai_context, "Dana Transfer ke Daerah (TKD)")


# --------------------------------------------------------------------------
# Router Halaman
# --------------------------------------------------------------------------
if menu == "1. Pagu & Realisasi":
    render_page_1()
elif menu == "2. Prioritas Presiden":
    render_page_2()
elif menu == "3. Program Strategis":
    render_page_3()
elif menu == "4. Transfer ke Daerah":
    render_page_4()