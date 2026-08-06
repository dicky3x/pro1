"""
common.py
----------
Modul bersama dipakai oleh Halaman 1 (app.py), Halaman 2 (pages/2_*.py), dan
Halaman 3 (pages/3_*.py): loading data, login, format kode, proyeksi, dan helper Groq/KPI.
"""

import os
import subprocess
from datetime import date, datetime as _dt

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from groq import Groq

# --------------------------------------------------------------------------
# Konstanta umum
# --------------------------------------------------------------------------

BULAN_KOLOM = ["JAN", "FEB", "MAR", "APR", "MEI", "JUN",
               "JUL", "AGS", "SEP", "OKT", "NOV", "DES"]
BULAN_LABEL = {i + 1: b for i, b in enumerate(BULAN_KOLOM)}

# Label jenis belanja versi singkat -- menggantikan label panjang bawaan data sumber.
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

# Kode jenis belanja yang termasuk kategori Transfer ke Daerah (dipakai Halaman 3).
KODE_JENIS_TKD = [61, 62, 63, 64, 65, 66]

# Label jenis belanja 51 (Belanja Pegawai) -- dipakai di beberapa tempat karena kategori ini
# punya perlakuan khusus: rumus proyeksi berbeda & TIDAK dibatasi maksimal pagu (lihat
# hitung_proyeksi_per_kategori & isi_tabel_proyeksi).
LABEL_BELANJA_PEGAWAI = LABEL_JENIS_BELANJA_SINGKAT[51]

# --- Penyesuaian khusus: kenaikan tunjangan kinerja Kementerian Pertahanan ---
# Mulai Juli 2026 ada kenaikan tukin (akun 512411) sebesar 28%/bulan, dirapel di
# September utk bulan Juli-Agustus yang sudah terlanjur direalisasikan sebelum kenaikan
# berlaku. Lihat sesuaikan_proyeksi_tukin_kemenhan() di bawah -- HANYA menyentuh bulan
# yang masih proyeksi (belum berakhir), tidak pernah mengubah angka realisasi aktual.
KDDEPT_KEMENHAN = 12
AKUN_TUKIN_KEMENHAN = [
    "Belanja Pegawai (Tunjangan Khusus/Kegiatan/Kinerja)",
    "Belanja Pegawai Tunjangan Khusus/Kegiatan/Kinerja PPPK",
]
BULAN_MULAI_KENAIKAN_TUKIN = 7  # Juli
PERSEN_KENAIKAN_TUKIN = 0.28
BULAN_RAPEL_TUKIN = 9  # September

GROQ_MODEL = os.environ.get("GROQ_MODEL", "openai/gpt-oss-120b")
# openai/gpt-oss-120b kadang tidak stabil untuk tool/function calling di Groq (error
# "tool_use_failed" / 400 BadRequestError sudah dilaporkan komunitas Groq). Kalau panggilan
# ber-tool gagal, dicoba ulang sekali pakai model cadangan ini.
GROQ_MODEL_FALLBACK_TOOLS = os.environ.get("GROQ_MODEL_FALLBACK_TOOLS", "moonshotai/kimi-k2-instruct-0905")

BOBOT_TAHUN = {1: 0.50, 2: 0.25, 3: 0.125, 4: 0.0625, 5: 0.0625}


def fmt_satker(kode) -> str:
    """Format kode satker jadi 6 digit dengan angka 0 di depan kalau kurang dari 6 digit."""
    if kode is None:
        return ""
    try:
        return f"{int(kode):06d}"
    except (TypeError, ValueError):
        return str(kode)


def fmt_dept(kode) -> str:
    """Format kode kementerian/lembaga jadi 3 digit dengan angka 0 di depan kalau kurang dari 3 digit."""
    if kode is None:
        return ""
    try:
        return f"{int(kode):03d}"
    except (TypeError, ValueError):
        return str(kode)


# --------------------------------------------------------------------------
# Load data -- dashboard pagu/realisasi satker (Halaman 1 & 3)
# --------------------------------------------------------------------------

@st.cache_data(show_spinner="Memuat data...")
def load_data_from_csv(path: str) -> pd.DataFrame:
    return pd.read_csv(path)


@st.cache_data(show_spinner="Memuat data dari Supabase...")
def load_data_from_supabase(url: str, key: str, table: str) -> pd.DataFrame:
    from supabase import create_client

    client = create_client(url, key)
    all_rows, page, page_size = [], 0, 1000
    while True:
        resp = (
            client.table(table)
            .select("*")
            .range(page * page_size, (page + 1) * page_size - 1)
            .execute()
        )
        rows = resp.data
        all_rows.extend(rows)
        if len(rows) < page_size:
            break
        page += 1
    return pd.DataFrame(all_rows)


def load_data() -> pd.DataFrame:
    use_supabase = st.secrets.get("USE_SUPABASE", "false") == "true" if hasattr(st, "secrets") else False
    if use_supabase:
        return load_data_from_supabase(
            st.secrets["SUPABASE_URL"],
            st.secrets["SUPABASE_KEY"],
            st.secrets.get("SUPABASE_TABLE", "pagu_realisasi"),
        )
    return load_data_from_csv("data/pagu_realisasi.csv.gz")


@st.cache_data(show_spinner=False)
def tanggal_update_data(path_csv: str = "data/pagu_realisasi.csv.gz") -> str:
    """Tanggal 'data terakhir diperbarui', diambil dari tanggal commit git terakhir yang
    mengubah file data ini di GitHub. Fallback ke waktu modifikasi file di disk kalau bukan repo git."""
    nama_bulan = ["Januari", "Februari", "Maret", "April", "Mei", "Juni",
                  "Juli", "Agustus", "September", "Oktober", "November", "Desember"]

    def _format(dt):
        return f"{dt.day} {nama_bulan[dt.month - 1]} {dt.year}, {dt.strftime('%H:%M')}"

    try:
        hasil = subprocess.run(
            ["git", "log", "-1", "--format=%cI", "--", path_csv],
            capture_output=True, text=True, timeout=5,
        )
        tanggal_str = hasil.stdout.strip()
        if tanggal_str:
            return _format(_dt.fromisoformat(tanggal_str))
    except Exception:
        pass

    try:
        return _format(_dt.fromtimestamp(os.path.getmtime(path_csv))) + " (perkiraan)"
    except Exception:
        return "tidak diketahui"


KOLOM_TEKS_CARI = [
    "NMDEPT", "NMSATKER", "PROVINSI", "KABKOTA", "FUNGSI", "SUBFUNGSI",
    "PROGRAM", "KEGIATAN", "OUTPUT", "AKUN",
]


@st.cache_data(show_spinner="Menyiapkan data...")
def siapkan_data(df_mentah: pd.DataFrame) -> pd.DataFrame:
    d = df_mentah.copy()
    # REALISASI & SISA PAGU dihitung ulang dari total kolom bulanan (JAN..DES) supaya semua
    # angka konsisten dengan rincian bulanannya.
    d["REALISASI"] = d[BULAN_KOLOM].sum(axis=1)
    d["SISA PAGU"] = d["PAGU"] - d["REALISASI"]

    d["LABEL_JENIS_BELANJA"] = (
        d["JENIS BELANJA"].map(LABEL_JENIS_BELANJA_SINGKAT).fillna(d["LABEL_JENIS_BELANJA"])
    )

    kolom_ada = [c for c in KOLOM_TEKS_CARI if c in d.columns]
    if kolom_ada:
        d["_TEKS_CARI"] = (
            d[kolom_ada].fillna("").astype(str).agg(" ".join, axis=1).str.lower()
        )
    else:
        d["_TEKS_CARI"] = ""
    return d


@st.cache_resource(show_spinner=False)
def get_data() -> pd.DataFrame:
    """Data utama (pagu/realisasi satker), sudah lewat siapkan_data(). Dipanggil sekali,
    dipakai bersama oleh Halaman 1 & 3 (cache_resource supaya objeknya sama & tidak
    di-copy ulang tiap halaman dibuka)."""
    return siapkan_data(load_data())


# --------------------------------------------------------------------------
# Login -- dipakai semua halaman (SATU sesi login berlaku utk semua halaman)
# --------------------------------------------------------------------------
# Username & password = kode satker masing-masing. Ada satu super user (kanwil04/admin)
# yang bisa melihat seluruh data semua satker.

SUPERUSER_USERNAME = "kanwil04"
SUPERUSER_PASSWORD = "admin"


def _cek_login(username: str, password: str, df_all: pd.DataFrame):
    username = (username or "").strip()
    password = (password or "").strip()
    if username == SUPERUSER_USERNAME and password == SUPERUSER_PASSWORD:
        return {"role": "super", "kdsatker": None}
    if username and username == password and username.isdigit():
        kdsatker = int(username)
        if kdsatker in df_all["KDSATKER"].unique():
            return {"role": "satker", "kdsatker": kdsatker}
    return None


def require_login(df_all: pd.DataFrame, judul_halaman: str = "Dashboard"):
    """Panggil di awal SETIAP halaman. Menampilkan form login & menghentikan halaman
    (st.stop()) kalau belum login, atau menampilkan status login + tombol logout di
    sidebar kalau sudah. Return dict auth ({'role':..., 'kdsatker':...})."""
    if "auth" not in st.session_state:
        st.session_state.auth = None

    if st.session_state.auth is None:
        # Saat belum login, app.py berhenti (st.stop()) SEBELUM sempat memanggil
        # st.navigation(pages) -- akibatnya Streamlit menampilkan navigasi bawaan
        # (daftar nama file di folder pages/) di sidebar. CSS ini menyembunyikan
        # navigasi bawaan itu supaya sidebar bersih selama di layar login.
        st.markdown(
            "<style>[data-testid='stSidebarNav'] {display: none;}</style>",
            unsafe_allow_html=True,
        )
        st.title(f"🔐 Login {judul_halaman}")
        st.caption(
            "Login memakai kode satker Anda sebagai username maupun password. Setelah login, "
            "Anda hanya bisa melihat data satker Anda sendiri di semua halaman dashboard."
        )
        with st.form("form_login"):
            username_input = st.text_input("Username (kode satker, 6 digit)", placeholder="mis. 012345")
            password_input = st.text_input("Password", type="password")
            submit_login = st.form_submit_button("Login")
        if submit_login:
            hasil_login = _cek_login(username_input, password_input, df_all)
            if hasil_login:
                st.session_state.auth = hasil_login
                st.rerun()
            else:
                st.error("Username atau password salah, atau kode satker tidak ditemukan di data.")
        st.stop()

    auth = st.session_state.auth
    is_super = auth["role"] == "super"

    with st.sidebar:
        if is_super:
            st.success(f"👤 Super User ({SUPERUSER_USERNAME})")
        else:
            st.success(f"👤 Satker: {fmt_satker(auth['kdsatker'])}")
        if st.button("🚪 Logout"):
            st.session_state.auth = None
            st.rerun()

    return auth


# --------------------------------------------------------------------------
# Proyeksi -- HYBRID FORECASTING: weighted historical monthly profile per
# (Satker x Jenis Belanja x Akun) + rolling forecast + fallback satker sejenis +
# confidence score & early warning tersedia sbg fungsi tambahan di bawah.
#
# Perbaikan dari versi sebelumnya:
#   1. Profil & tingkat serapan historis kini dihitung di granularitas SATKER x JENIS
#      BELANJA x AKUN dulu (sesuai spesifikasi), baru dijumlahkan ke scope filter yang
#      diminta -- bukan dihitung langsung di level scope teragregasi. Satker/akun tanpa
#      histori sendiri otomatis fallback ke profil satker sejenis (KDDEPT + JENIS
#      BELANJA + kelompok besar pagu), lalu ke rata-rata nasional per JENIS BELANJA.
#   2. isi_tabel_proyeksi: "jaring pengaman terakhir" DIPERBAIKI supaya hanya menyentuh
#      SATU bulan (bulan berjalan yang datanya baru sebagian), bukan seluruh sisa
#      tahun -- versi lama bisa membuat kategori yang seharusnya dibatasi pagu (mis.
#      Belanja Modal) tetap tampil melebihi pagu kalau data mentah kebetulan sudah
#      terisi di bulan-bulan yang harusnya masih "masa depan".
#   3. sesuaikan_proyeksi_tukin_kemenhan: pencocokan AKUN_TUKIN_KEMENHAN kini
#      menghilangkan whitespace di awal/akhir sebelum dibandingkan -- versi lama
#      kehilangan ~40% baris akun tukin yang punya spasi ekstra di data sumber,
#      sehingga baseline kenaikan tukin dihitung dari data yang tidak lengkap.
#
# Signature & makna return SETIAP fungsi publik (hitung_proyeksi_agregat,
# hitung_proyeksi_per_kategori, isi_tabel_proyeksi, sesuaikan_proyeksi_tukin_kemenhan,
# hitung_bulan_penuh_terakhir) TETAP SAMA seperti sebelumnya -- halaman-halaman yang
# memanggilnya (app.py, pages/2_*.py, pages/3_*.py) TIDAK PERLU diubah.
# --------------------------------------------------------------------------

def _filter_entitas(df_all: pd.DataFrame, thn: int, filter_dict: dict) -> pd.DataFrame:
    d = df_all[df_all["TAHUN"] == thn]
    for kolom, nilai in filter_dict.items():
        if nilai is not None:
            d = d[d[kolom] == nilai]
    return d




# ============================================================================
# Konfigurasi tambahan utk hybrid forecasting granular
# ============================================================================

KUNCI_GRANULAR = ["KDSATKER", "JENIS BELANJA", "AKUN"]
_EDGE_PAGU_BUCKET = [0, 100e6, 500e6, 1e9, 5e9, 1e10, np.inf]
_LABEL_PAGU_BUCKET = ["<100jt", "100jt-500jt", "500jt-1M", "1M-5M", "5M-10M", ">10M"]


def _bucket_pagu(pagu: pd.Series) -> pd.Series:
    """Kelompok ukuran pagu -- dipakai utk mencari 'satker sejenis' saat satker/akun
    tidak punya histori sendiri."""
    return pd.cut(pagu.clip(lower=0), bins=_EDGE_PAGU_BUCKET, labels=_LABEL_PAGU_BUCKET, include_lowest=True)


def _bobot_per_tahun_absolut(tahun_y: int, bobot_tahun_offset: dict) -> dict:
    """BOBOT_TAHUN aslinya berkunci offset (1=tahun_y-1, ..., 5=tahun_y-5). Fungsi ini
    mengubahnya jadi dict {tahun_absolut: bobot} supaya bisa dipakai langsung sbg
    map/groupby key thd kolom TAHUN."""
    return {tahun_y - i: b for i, b in bobot_tahun_offset.items()}


def _profil_granular(df_hist: pd.DataFrame, key_cols: list, bobot_tahun: dict,
                      bulan_kolom: list) -> pd.DataFrame:
    """Profil bulanan (proporsi, total=1), tingkat serapan historis tertimbang
    (RATE_HIST = realisasi/pagu), dan rerata REALISASI RUPIAH BULANAN tertimbang
    langsung (RP_<bulan> -- dipakai khusus Belanja Pegawai, lihat
    _hitung_proyeksi_belanja_pegawai_granular), per kombinasi key_cols.

    Weighted average dgn bobot_tahun, DINORMALISASI ULANG per key terhadap tahun yang
    datanya benar-benar tersedia (kalau suatu tahun histori kosong utk key tertentu,
    bobot tahun itu diabaikan & sisa bobot yang ada dinormalisasi -- sama seperti
    perilaku kode lama).
    """
    df = df_hist.copy()
    df["_TOTAL"] = df[bulan_kolom].sum(axis=1)
    df["_BOBOT"] = df["TAHUN"].map(bobot_tahun).fillna(0.0)

    # -- (a) profil bulanan: proporsi tiap bulan thd total tahun tsb --
    valid = df[(df["_TOTAL"] > 0) & (df["_BOBOT"] > 0)].copy()
    kolom_share = [f"_s_{b}" for b in bulan_kolom]
    for b, sc in zip(bulan_kolom, kolom_share):
        valid[sc] = (valid[b] / valid["_TOTAL"]) * valid["_BOBOT"]
    agg_share = valid.groupby(key_cols)[kolom_share].sum()
    bobot_share = valid.groupby(key_cols)["_BOBOT"].sum()
    profil = agg_share.div(bobot_share, axis=0)
    profil.columns = bulan_kolom

    # -- (b) rerata REALISASI RUPIAH BULANAN langsung (bukan proporsi) --
    dfb = df[df["_BOBOT"] > 0].copy()
    kolom_rp = [f"RP_{b}" for b in bulan_kolom]
    for b, rc in zip(bulan_kolom, kolom_rp):
        dfb[rc] = dfb[b] * dfb["_BOBOT"]
    agg_rp = dfb.groupby(key_cols)[kolom_rp].sum()
    bobot_rp = dfb.groupby(key_cols)["_BOBOT"].sum()
    rupiah_langsung = agg_rp.div(bobot_rp, axis=0)
    rupiah_langsung.columns = kolom_rp

    # -- (c) tingkat serapan historis (RATE_HIST = realisasi/pagu), tertimbang --
    df["_RATE"] = np.where(df["PAGU"] > 0, df["_TOTAL"] / df["PAGU"], np.nan)
    validr = df.dropna(subset=["_RATE"]).copy()
    validr["_WRATE"] = validr["_RATE"] * validr["_BOBOT"]
    agg = validr.groupby(key_cols).agg(
        _WRATE_SUM=("_WRATE", "sum"), _BOBOT_RATE=("_BOBOT", "sum"),
        RATE_MEAN=("_RATE", "mean"), RATE_STD=("_RATE", "std"), N_TAHUN=("_RATE", "count"),
    )
    agg["RATE_HIST"] = agg["_WRATE_SUM"] / agg["_BOBOT_RATE"].replace(0, np.nan)
    agg["CV_RATE"] = (agg["RATE_STD"] / agg["RATE_MEAN"].replace(0, np.nan)).abs()

    hasil = profil.join(rupiah_langsung, how="outer").join(
        agg[["RATE_HIST", "CV_RATE", "N_TAHUN"]], how="outer"
    )
    hasil["N_TAHUN"] = hasil["N_TAHUN"].fillna(0).astype(int)
    return hasil


def _gabungkan_fallback_granular(df_now: pd.DataFrame, df_hist: pd.DataFrame, bobot_tahun: dict,
                                  bulan_kolom: list,
                                  key_cols: list = KUNCI_GRANULAR,
                                  kelompok_cols: list = ("KDDEPT", "JENIS BELANJA", "_BUCKET_PAGU")
                                  ) -> pd.DataFrame:
    """Lengkapi tiap baris df_now (satker-jenis-akun tahun berjalan) dgn profil bulanan
    & RATE_HIST/RP_* -- prioritas: profil satker itu sendiri -> profil kelompok satker
    sejenis (KDDEPT+JENIS BELANJA+kelompok pagu) -> rata-rata nasional per JENIS BELANJA."""
    df_now = df_now.copy()
    df_now["_BUCKET_PAGU"] = _bucket_pagu(df_now["PAGU"])
    df_hist = df_hist.copy()
    df_hist["_BUCKET_PAGU"] = _bucket_pagu(df_hist["PAGU"])

    kolom_rp = [f"RP_{b}" for b in bulan_kolom]
    kolom_profil = list(bulan_kolom) + kolom_rp + ["RATE_HIST", "CV_RATE", "N_TAHUN"]

    profil_own = _profil_granular(df_hist, key_cols, bobot_tahun, bulan_kolom).reset_index()
    profil_grp = _profil_granular(df_hist, list(kelompok_cols), bobot_tahun, bulan_kolom).reset_index()
    profil_glb = _profil_granular(df_hist, ["JENIS BELANJA"], bobot_tahun, bulan_kolom).reset_index()

    basis = df_now.drop(columns=[c for c in bulan_kolom if c in df_now.columns])

    m_own = basis.merge(profil_own, on=key_cols, how="left")
    ada_sendiri = m_own["RATE_HIST"].notna() & (m_own["N_TAHUN"].fillna(0) > 0)

    m_grp = basis.merge(profil_grp, on=list(kelompok_cols), how="left")
    m_glb = basis.merge(profil_glb, on=["JENIS BELANJA"], how="left")

    hasil = m_own.copy()
    pakai_grp = (~ada_sendiri) & m_grp["RATE_HIST"].notna()
    for c in kolom_profil:
        hasil.loc[pakai_grp, c] = m_grp.loc[pakai_grp, c].values
    pakai_glb = (~ada_sendiri) & (~pakai_grp)
    for c in kolom_profil:
        hasil.loc[pakai_glb, c] = m_glb.loc[pakai_glb, c].values

    hasil["SUMBER_PROFIL"] = np.select(
        [ada_sendiri, pakai_grp, pakai_glb],
        ["Satker sendiri", "Kelompok sejenis (K/L+Jenis+Pagu)", "Rata-rata nasional per Jenis Belanja"],
        default="Tidak ada profil",
    )

    tanpa_profil = hasil["RATE_HIST"].isna() & hasil[kolom_rp[0]].isna()
    if tanpa_profil.any():
        for b, c in zip(bulan_kolom, kolom_rp):
            hasil.loc[tanpa_profil, b] = 1.0 / len(bulan_kolom)
            hasil.loc[tanpa_profil, c] = 0.0
        hasil.loc[tanpa_profil, "RATE_HIST"] = 1.0
        hasil.loc[tanpa_profil, "CV_RATE"] = 2.0
        hasil.loc[tanpa_profil, "N_TAHUN"] = 0
        hasil.loc[tanpa_profil, "SUMBER_PROFIL"] = "Tidak ada histori sama sekali"

    hasil["CV_RATE"] = hasil["CV_RATE"].fillna(0.0)

    # Pengaman terakhir: kombinasi yang PAGU-nya ada tapi realisasi historisnya nol
    # persis di semua tahun yang dipakai (mis. akun dialokasikan tapi tak pernah
    # dibelanjakan) punya RATE_HIST valid (=0) tapi profil bentuk bulanan (0/0) jadi
    # NaN -- isi dgn proporsi rata (1/12) & RP_* = 0 supaya tidak merembet jadi NaN
    # di baseline forecast.
    baris_share_kosong = hasil[bulan_kolom[0]].isna()
    if baris_share_kosong.any():
        for b, c in zip(bulan_kolom, kolom_rp):
            hasil.loc[baris_share_kosong, b] = 1.0 / len(bulan_kolom)
            hasil.loc[baris_share_kosong, c] = hasil.loc[baris_share_kosong, c].fillna(0.0)
        hasil["RATE_HIST"] = hasil["RATE_HIST"].fillna(0.0)

    return hasil


def _baseline_forecast_granular(gabung: pd.DataFrame, bulan_kolom: list, kolom_kategori: str = None) -> np.ndarray:
    """Hitung baseline forecast 12-bulan (rupiah, full tahun, BELUM direkonsiliasi dgn
    realisasi aktual -- rekonsiliasi itu tugas isi_tabel_proyeksi) per baris granular,
    lalu jumlahkan jadi satu vektor 12 -- utk dipakai hitung_proyeksi_agregat.

    Belanja Pegawai (LABEL_BELANJA_PEGAWAI) pakai RP_<bulan> (rerata rupiah bulanan
    historis langsung, TIDAK diskalakan pagu). Kategori lain pakai profil-share x
    RATE_HIST x PAGU (baris) -- setara dgn tingkat serapan historis x pagu tahun ini,
    disebar sesuai bentuk historis bulanan.
    """
    is_pegawai = gabung["JENIS BELANJA"] == 51  # kode jenis belanja Belanja Pegawai
    baseline = np.zeros((len(gabung), len(bulan_kolom)))
    for i, b in enumerate(bulan_kolom):
        rp_pegawai = gabung[f"RP_{b}"].to_numpy()
        non_pegawai = gabung[b].to_numpy() * gabung["RATE_HIST"].to_numpy() * gabung["PAGU"].to_numpy()
        baseline[:, i] = np.where(is_pegawai.to_numpy(), rp_pegawai, non_pegawai)
    return baseline


def _agregasi_ke_kunci_granular(df: pd.DataFrame, key_cols: list = KUNCI_GRANULAR,
                                 bulan_kolom: list = None) -> pd.DataFrame:
    """Data sumber bisa punya lebih dari satu baris utk kombinasi (TAHUN + KDSATKER +
    JENIS BELANJA + AKUN) yang sama (mis. beda program/kegiatan/sumber dana). Semua
    fungsi proyeksi granular di bawah ini butuh 1 baris = 1 deret waktu per kombinasi
    itu -- jadi dijumlahkan dulu di sini. Dipanggil otomatis di awal
    hitung_proyeksi_agregat & hitung_proyeksi_per_kategori."""
    bulan_kolom = bulan_kolom or BULAN_KOLOM
    kolom_id = [c for c in ("NMSATKER", "KDDEPT", "NMDEPT", "PROVINSI", "LABEL_JENIS_BELANJA") if c in df.columns]
    agg = {c: "sum" for c in list(bulan_kolom) + ["PAGU"]}
    for c in kolom_id:
        agg[c] = "first"
    return df.groupby(["TAHUN"] + list(key_cols), as_index=False).agg(agg)


def hitung_proyeksi_agregat(df_all: pd.DataFrame, tahun_y: int, pagu_y: float, filter_dict: dict):
    """[GRANULAR] Proyeksi 12 bulan (rupiah) utk seluruh entitas terpilih.

    Beda dgn versi lama: rate/profil dihitung per (KDSATKER, JENIS BELANJA, AKUN) dulu
    (dgn fallback satker sejenis kalau perlu), baru dijumlahkan ke scope filter_dict --
    bukan dihitung langsung di level scope teragregasi. Return (array atau None,
    daftar tahun dipakai), signature & makna return SAMA seperti versi lama.
    """
    df_all = _agregasi_ke_kunci_granular(df_all)
    bobot_tahun = _bobot_per_tahun_absolut(tahun_y, BOBOT_TAHUN)
    df_now = _filter_entitas(df_all, tahun_y, filter_dict)
    if df_now.empty:
        return None, []

    tahun_hist_kandidat = sorted(bobot_tahun.keys())
    df_hist = df_all[df_all["TAHUN"].isin(tahun_hist_kandidat)]
    for kolom, nilai in filter_dict.items():
        if nilai is not None:
            df_hist = df_hist[df_hist[kolom] == nilai]
    if df_hist.empty:
        return None, []

    tahun_dipakai = sorted(df_hist["TAHUN"].unique().tolist(), reverse=True)

    gabung = _gabungkan_fallback_granular(df_now, df_hist, bobot_tahun, BULAN_KOLOM)
    if (gabung["SUMBER_PROFIL"] == "Tidak ada histori sama sekali").all():
        return None, []

    baseline = _baseline_forecast_granular(gabung, BULAN_KOLOM)
    return baseline.sum(axis=0), tahun_dipakai


def hitung_proyeksi_per_kategori(df_all: pd.DataFrame, tahun_y: int, pagu_per_kategori_now: pd.Series,
                                  filter_dict: dict, kolom_kategori: str):
    """[GRANULAR] Proyeksi 12 bulan (rupiah) per kategori. Return (dict label->array(12)
    atau None, daftar tahun dipakai) -- signature & makna return SAMA seperti versi lama,
    supaya isi_tabel_proyeksi & pemanggil lain tidak perlu berubah.
    """
    hasil = {}
    tahun_dipakai_semua = set()
    df_all = _agregasi_ke_kunci_granular(df_all)
    bobot_tahun = _bobot_per_tahun_absolut(tahun_y, BOBOT_TAHUN)
    df_now = _filter_entitas(df_all, tahun_y, filter_dict)
    tahun_hist_kandidat = sorted(bobot_tahun.keys())
    df_hist_scope = df_all[df_all["TAHUN"].isin(tahun_hist_kandidat)]
    for kolom, nilai in filter_dict.items():
        if nilai is not None:
            df_hist_scope = df_hist_scope[df_hist_scope[kolom] == nilai]

    for label in pagu_per_kategori_now.index:
        d_now = df_now[df_now[kolom_kategori] == label]
        d_hist = df_hist_scope[df_hist_scope[kolom_kategori] == label]
        if d_now.empty or d_hist.empty:
            hasil[label] = None
            continue
        tahun_dipakai_semua.update(d_hist["TAHUN"].unique().tolist())

        gabung = _gabungkan_fallback_granular(d_now, d_hist, bobot_tahun, BULAN_KOLOM)
        if (gabung["SUMBER_PROFIL"] == "Tidak ada histori sama sekali").all():
            hasil[label] = None
            continue
        baseline = _baseline_forecast_granular(gabung, BULAN_KOLOM)
        hasil[label] = baseline.sum(axis=0)

    return hasil, sorted(tahun_dipakai_semua, reverse=True)


def isi_tabel_proyeksi(
    tabel_aktual: pd.DataFrame, proyeksi_per_kategori: dict, pagu_per_kategori: pd.Series,
    bulan_penuh_terakhir: int, kategori_dikecualikan_cap: str = LABEL_BELANJA_PEGAWAI,
    kategori_rupiah_langsung: frozenset = frozenset({LABEL_BELANJA_PEGAWAI}),
) -> pd.DataFrame:
    """Isi bulan-bulan setelah bulan_penuh_terakhir dengan proyeksi (bukan double-count -- lihat
    catatan di app.py), lalu cap total (aktual+proyeksi) maksimal 100% pagu utk kategori selain
    yang dikecualikan (default: Belanja Pegawai, karena satu-satunya yang boleh >100%)."""
    tabel_tampil = tabel_aktual.copy()
    if bulan_penuh_terakhir < 12:
        for kat in tabel_tampil.index:
            proyeksi_kat = proyeksi_per_kategori.get(kat)
            actual_sum = (
                tabel_tampil.loc[kat, BULAN_KOLOM[:bulan_penuh_terakhir]].sum()
                if bulan_penuh_terakhir else 0
            )
            if proyeksi_kat is None:
                # Tidak ada histori sama sekali -> fallback rata-rata realisasi tahun berjalan
                rerata_kat = actual_sum / bulan_penuh_terakhir if bulan_penuh_terakhir else 0
                bentuk_depan = np.full(12 - bulan_penuh_terakhir, rerata_kat)
                target_tahun_penuh = rerata_kat * 12
            else:
                # Bentuk sebaran bulan-bulan sisa mengikuti pola historis (rerata tertimbang
                # 50%-25%-12,5%-6,25%-6,25% dari tahun y-1..y-5), BUKAN dibagi rata -- supaya
                # bulan yang secara historis realisasinya tinggi (mis. akhir tahun) tetap
                # diproyeksikan tinggi, bukan disamaratakan.
                bentuk_depan = proyeksi_kat[bulan_penuh_terakhir:]
                target_tahun_penuh = proyeksi_kat.sum()

                if kat in kategori_rupiah_langsung:
                    # KHUSUS kategori rupiah-langsung (Belanja Pegawai): realisasi tahun
                    # berjalan bisa saja sudah melebihi target historis (mis. karena pagu
                    # tahun ini beda dari tahun-tahun sebelumnya, atau ada kenaikan gaji/
                    # tunjangan). Di kondisi itu target dinaikkan ke proyeksi run-rate
                    # (rata-rata realisasi bulan berjalan x 12) sbg lantai bawah -- TAPI
                    # bentuk sebaran bulanannya tetap dari pola historis, bukan rata.
                    # TIDAK diterapkan ke kategori lain krn realisasi tahunnya historis
                    # jarang mendekati 100% pagu, jadi run-rate disini biasanya cuma bikin
                    # proyeksi kebablasan ke atas (ke-cap 100% padahal seharusnya tidak).
                    rerata_kat = actual_sum / bulan_penuh_terakhir if bulan_penuh_terakhir else 0
                    target_runrate = rerata_kat * 12
                    target_tahun_penuh = max(target_tahun_penuh, target_runrate)

            sisa_target = max(target_tahun_penuh - actual_sum, 0)
            total_bentuk_depan = bentuk_depan.sum()
            if total_bentuk_depan > 0:
                proyeksi_depan = bentuk_depan * (sisa_target / total_bentuk_depan)
            elif sisa_target > 0:
                proyeksi_depan = np.full(12 - bulan_penuh_terakhir, sisa_target / (12 - bulan_penuh_terakhir))
            else:
                proyeksi_depan = np.zeros(12 - bulan_penuh_terakhir)
            for idx, m in enumerate(range(bulan_penuh_terakhir, 12)):
                tabel_tampil.loc[kat, BULAN_KOLOM[m]] = proyeksi_depan[idx]

    if bulan_penuh_terakhir < 12:
        for kat in tabel_tampil.index:
            if kat == kategori_dikecualikan_cap:
                continue
            pagu_kat = pagu_per_kategori.get(kat, 0)
            if not pagu_kat or pagu_kat <= 0:
                continue
            actual_sum = tabel_tampil.loc[kat, BULAN_KOLOM[:bulan_penuh_terakhir]].sum()
            proyeksi_depan = tabel_tampil.loc[kat, BULAN_KOLOM[bulan_penuh_terakhir:]]
            total_proyeksi_depan = proyeksi_depan.sum()
            sisa_pagu_kat = max(pagu_kat - actual_sum, 0)
            if total_proyeksi_depan > sisa_pagu_kat:
                faktor_skala = (sisa_pagu_kat / total_proyeksi_depan) if total_proyeksi_depan > 0 else 0
                tabel_tampil.loc[kat, BULAN_KOLOM[bulan_penuh_terakhir:]] = proyeksi_depan * faktor_skala

        # Jaring pengaman terakhir: kolom bulan >= bulan_penuh_terakhir TIDAK BOLEH lebih kecil
        # dari data aktual yang sudah benar-benar tercatat di sumber data (mis. bulan berjalan
        # yang datanya baru sebagian tapi sudah lebih besar dari estimasi proyeksinya). Tanpa
        # ini, "Total Realisasi + Proyeksi Akhir Tahun" bisa keliru tampil LEBIH KECIL dari
        # "Total Realisasi" murni -- yang secara logika tidak boleh terjadi.
        for kat in tabel_tampil.index:
            m = bulan_penuh_terakhir  # HANYA bulan berjalan (yg baru sebagian), bukan seluruh sisa tahun
            kol = BULAN_KOLOM[m]
            asli = tabel_aktual.loc[kat, kol]
            if pd.notna(asli) and asli > tabel_tampil.loc[kat, kol]:
                tabel_tampil.loc[kat, kol] = asli

    return tabel_tampil


def sesuaikan_proyeksi_tukin_kemenhan(
    df_scope: pd.DataFrame, tabel_tampil: pd.DataFrame, bulan_penuh_terakhir: int,
) -> pd.DataFrame:
    """Penyesuaian khusus proyeksi Belanja Pegawai utk Kementerian Pertahanan: mulai Juli
    2026 ada kenaikan tunjangan kinerja (akun 512411) sebesar 28%/bulan, dirapel di
    September utk Juli-Agustus yang sudah terlanjur direalisasikan sebelum kenaikan
    berlaku. HANYA menyentuh bulan yang masih proyeksi (>= bulan_penuh_terakhir); tidak
    pernah mengubah angka bulan yang sudah aktual/berakhir.

    df_scope: data mentah pada cakupan (satker/kementerian/semua) yang sedang aktif di
              dashboard -- fungsi ini otomatis menyaring baris Kemenhan (KDDEPT=12) dan
              tidak melakukan apa-apa kalau Kemenhan tidak ada di cakupan ini.
    """
    if bulan_penuh_terakhir >= 12 or LABEL_BELANJA_PEGAWAI not in tabel_tampil.index:
        return tabel_tampil

    tukin = df_scope[
        (df_scope["KDDEPT"] == KDDEPT_KEMENHAN)
        & (df_scope["AKUN"].astype(str).str.strip().isin([a.strip() for a in AKUN_TUKIN_KEMENHAN]))
    ]
    if tukin.empty:
        return tabel_tampil  # Kemenhan tidak ada di cakupan yang sedang dilihat

    # Baseline bulanan = rata-rata realisasi AKTUAL akun tukin sebelum bulan kenaikan
    # (Jan..Jun yang sudah benar-benar berakhir).
    bulan_baseline = [
        BULAN_KOLOM[i] for i in range(BULAN_MULAI_KENAIKAN_TUKIN - 1) if i < bulan_penuh_terakhir
    ]
    if not bulan_baseline:
        return tabel_tampil  # belum cukup data aktual utk hitung baseline
    baseline_bulanan = tukin[bulan_baseline].sum().sum() / len(bulan_baseline)
    kenaikan_bulanan = baseline_bulanan * PERSEN_KENAIKAN_TUKIN

    tabel_tampil = tabel_tampil.copy()
    for m in range(bulan_penuh_terakhir, 12):
        bulan_ke = m + 1
        if bulan_ke >= BULAN_MULAI_KENAIKAN_TUKIN:
            tabel_tampil.loc[LABEL_BELANJA_PEGAWAI, BULAN_KOLOM[m]] += kenaikan_bulanan
        if bulan_ke == BULAN_RAPEL_TUKIN:
            # Rapel utk bulan-bulan (Jul, Agst) yang SUDAH aktual (bukan proyeksi) sebelum
            # kenaikan sempat tercermin di realisasi -- bulan yg masih proyeksi sendiri
            # sudah otomatis dapat kenaikannya langsung dari baris di atas, tidak perlu rapel.
            bulan_tertunda = [b for b in (7, 8) if b <= bulan_penuh_terakhir]
            if bulan_tertunda:
                tabel_tampil.loc[LABEL_BELANJA_PEGAWAI, BULAN_KOLOM[m]] += (
                    kenaikan_bulanan * len(bulan_tertunda)
                )

    return tabel_tampil


def hitung_bulan_penuh_terakhir(df_entitas: pd.DataFrame, tahun_y: int) -> tuple:
    """Return (bulan_terakhir, bulan_penuh_terakhir) dari sebuah subset data yang sudah difilter tahun."""
    monthly = df_entitas[BULAN_KOLOM].sum()
    bulan_terisi = [i + 1 for i, v in enumerate(monthly.values) if v != 0]
    bulan_terakhir = max(bulan_terisi) if bulan_terisi else 0

    hari_ini = date.today()
    if tahun_y < hari_ini.year:
        bulan_penuh_terakhir = bulan_terakhir
    elif tahun_y > hari_ini.year:
        bulan_penuh_terakhir = 0
    else:
        bulan_penuh_terakhir = min(bulan_terakhir, hari_ini.month - 1)
    return bulan_terakhir, bulan_penuh_terakhir


# --------------------------------------------------------------------------
# Groq client & kartu KPI
# --------------------------------------------------------------------------

def get_groq_client():
    api_key = st.secrets.get("GROQ_API_KEY") if hasattr(st, "secrets") else None
    api_key = api_key or os.environ.get("GROQ_API_KEY")
    if not api_key:
        return None
    return Groq(api_key=api_key)


def _pesan_error_groq(e) -> str:
    """Gabungkan detail error Groq (message + body) jadi satu string lowercase utk deteksi
    jenis error, dipakai oleh _groq_chat_dgn_fallback & pesan error yang ditampilkan ke user."""
    detail = str(getattr(e, "message", None) or e)
    body = getattr(e, "body", None)
    return f"{detail} {body}".lower()


def _groq_chat_dgn_fallback(client, **kwargs):
    """Panggil Groq chat completion pakai GROQ_MODEL; kalau gagal spesifik karena masalah
    tool-calling (model tsb kadang tidak stabil utk tool/function calling di Groq -- error
    "tool_use_failed" / "Tool choice is none, but model called a tool"), otomatis dicoba
    ulang SEKALI pakai GROQ_MODEL_FALLBACK_TOOLS. Kalau tetap gagal atau errornya bukan soal
    tool-calling, exception aslinya dilempar lagi supaya ditangani pemanggil seperti biasa."""
    try:
        return client.chat.completions.create(model=GROQ_MODEL, **kwargs)
    except Exception as e:
        pesan = _pesan_error_groq(e)
        if "tool_use_failed" in pesan or "tool choice" in pesan:
            return client.chat.completions.create(model=GROQ_MODEL_FALLBACK_TOOLS, **kwargs)
        raise


# Batas ukuran hasil tool-calling yang dikirim balik ke Groq -- mencegah error "Please reduce
# the length of the messages or completion" kalau hasil query/agregasi user ternyata sangat
# besar (mis. banyak baris/kolom dgn teks panjang).
MAKS_BARIS_HASIL_TOOL = 50
MAKS_PANJANG_TEKS_SEL = 300
MAKS_PANJANG_JSON_TOOL = 8000


def _potong_teks_dalam_records(records: list) -> list:
    """Potong nilai teks yang kepanjangan di tiap field hasil query/agregasi, supaya hasil
    tool-calling yang dikirim ke Groq tidak membengkak gara-gara 1-2 kolom teks panjang."""
    hasil = []
    for r in records:
        baris = {}
        for k, v in r.items():
            if isinstance(v, str) and len(v) > MAKS_PANJANG_TEKS_SEL:
                v = v[:MAKS_PANJANG_TEKS_SEL] + "…(dipotong)"
            baris[k] = v
        hasil.append(baris)
    return hasil


def _json_tool_aman(hasil: dict) -> str:
    """Serialize hasil tool ke JSON dgn batas ukuran akhir sbg jaring pengaman terakhir."""
    import json as _json
    s = _json.dumps(hasil, ensure_ascii=False, default=str)
    if len(s) > MAKS_PANJANG_JSON_TOOL:
        s = s[:MAKS_PANJANG_JSON_TOOL] + '... (dipotong krn hasilnya terlalu besar -- coba pertanyaan yang lebih spesifik/sempit)'
    return s


def kpi_card(label: str, value: str, delta: str = None):
    delta_html = (
        f'<div style="font-size:0.85rem;color:#16a34a;margin-top:4px;">{delta}</div>'
        if delta else ""
    )
    st.markdown(
        f"""
        <div style="background:#f8fafc;border:1px solid #e2e8f0;border-radius:12px;
                    padding:16px 18px;min-height:110px;">
            <div style="font-size:0.85rem;color:#64748b;margin-bottom:6px;">{label}</div>
            <div style="font-size:clamp(1rem, 2.1vw, 1.6rem);font-weight:700;color:#0f172a;
                        white-space:normal;overflow-wrap:break-word;line-height:1.25;">{value}</div>
            {delta_html}
        </div>
        """,
        unsafe_allow_html=True,
    )


DISCLAIMER_TEKS = (
    "ℹ️ **Disclaimer**: data ini ditarik manual secara berkala melalui sintesa.kemenkeu.go.id "
    "kewenangan Kanwil DJPb Riau. Beberapa program strategis dan prioritas presiden di Riau "
    "mungkin tidak muncul di dashboard ini. Hal ini kemungkinan dikarenakan program strategis "
    "dan prioritas presiden tersebut didanai secara terpusat, tidak dibayarkan melalui KPPN di "
    "wilayah Riau."
)


def render_disclaimer():
    """Disclaimer standar yang tampil di setiap halaman dashboard (Halaman 1-4)."""
    st.caption(DISCLAIMER_TEKS)


def satker_ada_di_path(data_path: str, kdsatker: int) -> bool:
    """Cek apakah suatu kode satker punya baris data di file CSV kategori (Prioritas
    Presiden / Program Strategis) -- dipakai app.py utk sembunyikan halaman yang tidak
    relevan bagi satker yang sedang login."""
    if kdsatker is None or not os.path.exists(data_path):
        return False
    try:
        d = _load_csv_kategori(data_path)
    except Exception:
        return False
    return kdsatker in d["KDSATKER"].unique()


def buat_cari_generik(df_all: pd.DataFrame, kolom_teks: list, scope_kdsatker: int = None):
    """Bikin fungsi pencarian generik (dipakai tool-calling AI di chat box) yang mencari
    kata kunci di kolom_teks, opsional filter provinsi & tahun, dan opsional dibatasi ke
    satu KDSATKER saja (utk user satker biasa). Hasil dikelompokkan per satker.

    Return: fungsi cari(kata_kunci, provinsi=None, tahun_cari=None) -> dict
    """
    teks_gabungan = df_all[kolom_teks].fillna("").astype(str).agg(" ".join, axis=1).str.lower()

    def cari(kata_kunci: list = None, provinsi: str = None, tahun_cari: int = None) -> dict:
        d = df_all
        teks = teks_gabungan
        if scope_kdsatker is not None:
            mask = d["KDSATKER"] == scope_kdsatker
            d, teks = d[mask], teks[mask]
        if tahun_cari is not None and "TAHUN" in d.columns:
            mask = d["TAHUN"] == tahun_cari
            d, teks = d[mask], teks[mask]
        if provinsi and "PROVINSI" in d.columns:
            mask = d["PROVINSI"].str.contains(provinsi, case=False, na=False)
            d, teks = d[mask], teks[mask]
        if kata_kunci:
            mask = pd.Series(False, index=d.index)
            for kw in kata_kunci:
                mask = mask | teks.str.contains(str(kw).lower(), na=False)
            d = d[mask]

        if d.empty:
            return {
                "ditemukan": False,
                "pesan": "Tidak ada baris data yang cocok dengan kata kunci/filter ini.",
            }

        rincian = (
            d.groupby(["KDDEPT", "NMDEPT", "KDSATKER", "NMSATKER"])
            .agg(PAGU=("PAGU", "sum"), REALISASI=("REALISASI", "sum"))
            .reset_index().sort_values("PAGU", ascending=False).head(30)
        )
        rincian["KDSATKER"] = rincian["KDSATKER"].apply(fmt_satker)
        rincian["KDDEPT"] = rincian["KDDEPT"].apply(fmt_dept)

        return {
            "ditemukan": True,
            "jumlah_baris_cocok": int(len(d)),
            "jumlah_satker_ditemukan": int(rincian.shape[0]),
            "total_pagu": float(d["PAGU"].sum()),
            "total_realisasi": float(d["REALISASI"].sum()),
            "rincian_per_satker_top30": rincian.to_dict(orient="records"),
            "catatan": (
                "rincian_per_satker_top30 diurutkan dari pagu terbesar, dibatasi 30 baris teratas."
                + (" Pencarian dibatasi hanya data satker Anda sendiri." if scope_kdsatker is not None else "")
            ),
        }

    return cari


def render_ai_section(
    ringkasan_fn, cari_fn, page_key: str, narasi_variasi_key: str = "",
    deskripsi_tool: str = (
        "Mencari & menjumlahkan pagu/realisasi berdasarkan kata kunci tema/program/kegiatan, "
        "dan opsional filter provinsi atau tahun. WAJIB dipakai untuk pertanyaan yang menyebutkan "
        "tema, lokasi/provinsi tertentu, atau satker/kementerian yang BUKAN yang sedang aktif."
    ),
):
    """Render bagian 'Narasi Otomatis AI' + chat box tanya-AI (dgn tool-calling pencarian
    tematik). Dipakai semua halaman (app.py & pages/*.py) supaya seragam.

    ringkasan_fn: callable() -> str, ringkasan konteks data terkini utk system prompt
    cari_fn: callable(kata_kunci, provinsi=None, tahun_cari=None) -> dict, tool pencarian
             (lihat buat_cari_generik, atau fungsi custom seperti di app.py)
    page_key: id unik per halaman (mis. "satker", "prioritas", "strategis", "tkd") --
              dipakai memisahkan riwayat chat & key widget antar halaman
    narasi_variasi_key: id tambahan yang berubah kalau filter berubah (mis. tahun/satker)
              -- dipakai cache narasi supaya tidak nyampur antar filter berbeda
    """
    import json as _json

    client = get_groq_client()

    tools_groq = [{
        "type": "function",
        "function": {
            "name": "cari_anggaran",
            "description": deskripsi_tool,
            "parameters": {
                "type": "object",
                "properties": {
                    "kata_kunci": {
                        "type": "array", "items": {"type": "string"},
                        "description": "Daftar kata kunci (boleh beberapa sinonim) untuk dicari.",
                    },
                    "provinsi": {
                        "type": ["string", "null"],
                        "description": "Nama provinsi untuk filter lokasi (opsional, boleh null).",
                    },
                    "tahun_cari": {
                        "type": ["integer", "null"],
                        "description": "Tahun anggaran (opsional, boleh null utk pakai tahun yang sedang dipilih).",
                    },
                },
                "required": ["kata_kunci"],
            },
        },
    }]

    def _jalankan_tool_call(nama_fungsi, args):
        if nama_fungsi == "cari_anggaran":
            hasil = cari_fn(
                kata_kunci=args.get("kata_kunci", []),
                provinsi=args.get("provinsi"),
                tahun_cari=args.get("tahun_cari"),
            )
        else:
            hasil = {"error": f"Fungsi tidak dikenal: {nama_fungsi}"}
        return _json_tool_aman(hasil)

    # --- Narasi otomatis ---
    st.subheader("🤖 Narasi Otomatis")
    if client is None:
        st.info(
            "Narasi AI belum aktif. Tambahkan `GROQ_API_KEY` di file `.streamlit/secrets.toml` "
            "(lihat README.md) untuk mengaktifkan fitur ini."
        )
    else:
        cache_key = f"narasi_{page_key}_{narasi_variasi_key}"
        if st.button("Buat / Perbarui Narasi", key=f"btn_{cache_key}"):
            with st.spinner("AI sedang menyusun narasi..."):
                try:
                    resp = _groq_chat_dgn_fallback(
                        client,
                        messages=[
                            {
                                "role": "system",
                                "content": (
                                    "Kamu adalah analis anggaran pemerintah Indonesia. Tulis narasi "
                                    "singkat (1-2 paragraf, bahasa Indonesia formal) yang menjelaskan "
                                    "kondisi pagu dan realisasi anggaran berikut, termasuk kecukupan "
                                    "serapan. Jangan mengulang angka mentah berlebihan, fokus insight."
                                ),
                            },
                            {"role": "user", "content": ringkasan_fn()},
                        ],
                    )
                    st.session_state[cache_key] = resp.choices[0].message.content
                except Exception as e:
                    st.error(f"Gagal membuat narasi karena gangguan AI ({type(e).__name__}). Coba lagi.")

        if cache_key in st.session_state:
            st.write(st.session_state[cache_key])
        else:
            st.caption("Klik tombol di atas untuk membuat narasi.")

    st.divider()

    # --- Chat box ---
    chat_state_key = f"chat_history_{page_key}"
    if chat_state_key not in st.session_state:
        st.session_state[chat_state_key] = []

    MAKS_HISTORI_DIKIRIM = 6

    col_chat_title, col_chat_reset = st.columns([5, 1])
    with col_chat_title:
        st.subheader("💬 Tanya AI tentang Data Ini")
    with col_chat_reset:
        if st.button("🗑️ Reset Chat", key=f"resetchat_{page_key}"):
            st.session_state[chat_state_key] = []
            st.rerun()

    st.caption(
        "Bisa tanya soal data yang sedang ditampilkan, atau tema/lokasi lain -- AI otomatis "
        "mencari di data yang tersedia di halaman ini kalau perlu."
    )

    for msg in st.session_state[chat_state_key]:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])

    prompt = st.chat_input("Tulis pertanyaan tentang data ini...", key=f"chatinput_{page_key}")

    if prompt:
        st.session_state[chat_state_key].append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.write(prompt)

        if client is None:
            jawaban = "Fitur chat AI belum aktif karena GROQ_API_KEY belum di-set. Lihat README.md."
        else:
            with st.spinner("AI sedang menjawab..."):
                system_msg = {
                    "role": "system",
                    "content": (
                        "Kamu adalah asisten analisis anggaran pemerintah Indonesia. Untuk "
                        "pertanyaan tentang data yang SEDANG ditampilkan, jawab langsung memakai "
                        "data di bawah ini. Untuk pertanyaan tentang tema/lokasi/entitas LAIN, "
                        "WAJIB panggil fungsi cari_anggaran -- jangan mengarang angka. Kalau hasil "
                        "pencarian menyatakan tidak ditemukan, katakan jujur, jangan mengarang.\n\n"
                        + ringkasan_fn()
                    ),
                }
                histori_dikirim = st.session_state[chat_state_key][-MAKS_HISTORI_DIKIRIM:]
                messages = [system_msg] + histori_dikirim

                jawaban = None
                try:
                    resp = _groq_chat_dgn_fallback(
                        client, messages=messages, tools=tools_groq, tool_choice="auto",
                    )
                    msg = resp.choices[0].message

                    if msg.tool_calls:
                        messages.append({
                            "role": "assistant",
                            "content": msg.content or "",
                            "tool_calls": [
                                {
                                    "id": tc.id, "type": "function",
                                    "function": {"name": tc.function.name, "arguments": tc.function.arguments},
                                }
                                for tc in msg.tool_calls
                            ],
                        })
                        for tc in msg.tool_calls:
                            try:
                                args = _json.loads(tc.function.arguments)
                            except Exception:
                                args = {}
                            hasil_tool = _jalankan_tool_call(tc.function.name, args)
                            messages.append({"role": "tool", "tool_call_id": tc.id, "content": hasil_tool})
                        resp2 = _groq_chat_dgn_fallback(client, messages=messages)
                        jawaban = resp2.choices[0].message.content
                    else:
                        jawaban = msg.content
                except Exception as e:
                    pesan_lower = _pesan_error_groq(e)
                    detail = getattr(e, "message", None) or str(e)
                    body = getattr(e, "body", None)
                    if "reduce the length" in pesan_lower or "context" in pesan_lower and "length" in pesan_lower:
                        jawaban = (
                            "⚠️ Percakapan atau hasil pencariannya terlalu besar buat diproses AI. "
                            "Coba: (1) klik **🗑️ Reset Chat** untuk mulai percakapan baru, atau "
                            "(2) ajukan pertanyaan yang lebih spesifik/sempit (mis. sebutkan nama "
                            "kementerian/satker/kata kunci yang lebih pasti)."
                        )
                    elif "tool_use_failed" in pesan_lower or "tool choice" in pesan_lower:
                        jawaban = (
                            "⚠️ Model AI sedang tidak stabil untuk fitur pencarian data ini "
                            "(sudah dicoba dgn model cadangan, tetap gagal). Coba tanya ulang "
                            "beberapa saat lagi."
                        )
                    else:
                        jawaban = f"⚠️ Gagal memanggil Groq API: {detail}"
                    with st.expander("Detail error (untuk debugging)"):
                        st.code(f"{type(e).__name__}: {detail}\n\nBody: {body}")

        st.session_state[chat_state_key].append({"role": "assistant", "content": jawaban})
        with st.chat_message("assistant"):
            st.write(jawaban)


# --------------------------------------------------------------------------
# Dashboard generik utk data "kategori" (Prioritas Presiden / Program Strategis) --
# Halaman 2 & 3 sama-sama memanggil ini karena struktur sumber datanya identik, cuma
# beda kolom kategori & isinya. Lihat build_prioritas_strategis.py utk cara datanya dibuat.
#
# Beda penting dengan Halaman 1/4: file sumbernya HANYA berisi tahun 2026 (belum ada
# histori tahun sebelumnya), jadi hitung_proyeksi_agregat/per_kategori otomatis fallback
# ke metode rata-rata bulan berjalan (return None -> tidak ada histori yang bisa dipakai).
# --------------------------------------------------------------------------

@st.cache_data(show_spinner="Memuat data...")
def _load_csv_kategori(path: str) -> pd.DataFrame:
    d = pd.read_csv(path)
    d["REALISASI"] = d[BULAN_KOLOM].sum(axis=1)
    d["SISA PAGU"] = d["PAGU"] - d["REALISASI"]
    return d


def render_dashboard_kategori(
    data_path: str, judul_halaman: str, icon: str, label_kategori: str,
    scope_kdsatker: int = None, page_key: str = None,
):
    """Render dashboard lengkap (filter, KPI, grafik, tabel rincian, AI) untuk data
    prioritas/strategis. Dipanggil oleh pages/2_*.py (Prioritas Presiden) & pages/3_*.py
    (Program Strategis) dengan parameter berbeda.

    data_path: path file CSV (gzip) hasil olahan build_prioritas_strategis.py
    judul_halaman: judul ditampilkan di halaman, mis. "Dashboard Prioritas Presiden"
    icon: emoji ikon halaman
    label_kategori: nama kolom kategori utk ditampilkan di UI, mis. "Prioritas Presiden"
    scope_kdsatker: kalau diisi (user satker biasa), data dibatasi ke satker ini saja
    page_key: id unik halaman utk AI chat/narasi (default: label_kategori disederhanakan)
    """
    if not os.path.exists(data_path):
        st.warning(f"Halaman ini belum aktif -- file data belum tersedia di `{data_path}`.")
        st.stop()

    df = _load_csv_kategori(data_path)
    if scope_kdsatker is not None:
        df = df[df["KDSATKER"] == scope_kdsatker]
        if df.empty:
            st.info(
                f"Satker Anda tidak memiliki anggaran terkait {label_kategori.lower()}."
            )
            st.stop()

    page_key = page_key or label_kategori.lower().replace(" ", "_")

    # --- Sidebar filter ---
    st.sidebar.header("Filter")
    tahun_list = sorted(df["TAHUN"].unique(), reverse=True)
    tahun = st.sidebar.selectbox("Tahun", tahun_list)
    df_tahun = df[df["TAHUN"] == tahun]

    semua_kategori_label = f"— Semua {label_kategori} —"
    kategori_list = sorted(df_tahun["KATEGORI"].dropna().unique().tolist())
    kategori_pilih = st.sidebar.selectbox(label_kategori, [semua_kategori_label] + kategori_list)

    if kategori_pilih == semua_kategori_label:
        kategori = None
        df_scope = df_tahun
    else:
        kategori = kategori_pilih
        df_scope = df_tahun[df_tahun["KATEGORI"] == kategori]

    # --- Agregasi ---
    pagu_total = df_scope["PAGU"].sum()
    realisasi_total = df_scope["REALISASI"].sum()
    sisa_pagu = df_scope["SISA PAGU"].sum()
    persen_serapan = (realisasi_total / pagu_total * 100) if pagu_total else 0

    monthly = df_scope[BULAN_KOLOM].sum()
    kumulatif = monthly.cumsum()
    bulan_terakhir, bulan_penuh_terakhir = hitung_bulan_penuh_terakhir(df_scope, tahun)

    # --- Proyeksi (fallback otomatis krn data cuma 1 tahun -- lihat catatan di atas) ---
    filter_entitas = {"KATEGORI": kategori}
    proyeksi_agregat_bulanan, tahun_dipakai = hitung_proyeksi_agregat(df, tahun, pagu_total, filter_entitas)
    if proyeksi_agregat_bulanan is None:
        rerata_bulanan = (kumulatif.iloc[bulan_terakhir - 1] / bulan_terakhir) if bulan_terakhir else 0
        proyeksi_akhir_tahun = rerata_bulanan * 12
        metode_proyeksi = "fallback"
    else:
        target_tahun_penuh = proyeksi_agregat_bulanan.sum()
        proyeksi_akhir_tahun = max(realisasi_total, target_tahun_penuh)
        metode_proyeksi = "historis"
    persen_proyeksi = (proyeksi_akhir_tahun / pagu_total * 100) if pagu_total else 0

    # --- Header & KPI ---
    st.title(f"{icon} {judul_halaman}")
    st.caption(f"🕒 Data terakhir diperbarui: {tanggal_update_data(data_path)}")
    st.caption(f"{kategori or f'Semua {label_kategori}'} — Tahun {tahun}")
    render_disclaimer()

    r1c1, r1c2 = st.columns(2)
    r2c1, r2c2 = st.columns(2)
    with r1c1:
        kpi_card("Pagu", f"Rp {pagu_total:,.0f}")
    with r1c2:
        kpi_card("Realisasi", f"Rp {realisasi_total:,.0f}", f"{persen_serapan:.1f}% dari pagu")
    with r2c1:
        kpi_card("Sisa Pagu", f"Rp {sisa_pagu:,.0f}")
    with r2c2:
        kpi_card(
            "Proyeksi Realisasi Akhir Tahun",
            f"Rp {proyeksi_akhir_tahun:,.0f}",
            f"{persen_proyeksi:.1f}% dari pagu",
        )

    st.divider()

    # --- Grafik batang ---
    col1, col2 = st.columns([3, 2])
    with col1:
        st.subheader("Realisasi per Bulan")
        bar_df = pd.DataFrame({"Bulan": BULAN_KOLOM, "Realisasi": monthly.values})
        fig_bar = px.bar(bar_df, x="Bulan", y="Realisasi", text_auto=".2s")
        fig_bar.update_layout(yaxis_title="Rupiah", xaxis_title=None)
        st.plotly_chart(fig_bar, use_container_width=True)
    with col2:
        st.subheader("Pagu vs Realisasi")
        fig_pv = go.Figure(data=[
            go.Bar(name="Pagu", x=["Total"], y=[pagu_total]),
            go.Bar(name="Realisasi", x=["Total"], y=[realisasi_total]),
        ])
        fig_pv.update_layout(barmode="group", yaxis_title="Rupiah")
        st.plotly_chart(fig_pv, use_container_width=True)

    # --- Pie charts ---
    st.subheader("Komposisi")
    p1, p2 = st.columns(2)
    with p1:
        st.caption("Realisasi vs Sisa Pagu")
        fig_pie1 = px.pie(
            names=["Realisasi", "Sisa Pagu"], values=[realisasi_total, max(sisa_pagu, 0)], hole=0.4,
        )
        st.plotly_chart(fig_pie1, use_container_width=True)
    with p2:
        if kategori is None:
            st.caption(f"Realisasi per {label_kategori}")
            per_kat = (
                df_scope.groupby("KATEGORI")["REALISASI"].sum()
                .sort_values(ascending=False).reset_index()
            )
            fig_pie2 = px.pie(per_kat, names="KATEGORI", values="REALISASI", hole=0.4)
        else:
            st.caption("Realisasi per Kabupaten/Kota")
            per_wil = (
                df_scope.groupby("KABKOTA")["REALISASI"].sum()
                .sort_values(ascending=False).reset_index()
            )
            fig_pie2 = px.pie(per_wil, names="KABKOTA", values="REALISASI", hole=0.4)
        st.plotly_chart(fig_pie2, use_container_width=True)

    # --- Perbandingan antar kategori (hanya kalau "Semua" dipilih) ---
    if kategori is None:
        st.subheader(f"Perbandingan antar {label_kategori}")
        per_kat_tabel = (
            df_tahun.groupby("KATEGORI")
            .agg(Pagu=("PAGU", "sum"), Realisasi=("REALISASI", "sum"))
            .reset_index()
            .sort_values("Pagu", ascending=False)
        )
        per_kat_tabel["Persen Realisasi"] = (
            per_kat_tabel["Realisasi"] / per_kat_tabel["Pagu"].replace(0, np.nan) * 100
        ).fillna(0)

        fig_kat = go.Figure(data=[
            go.Bar(name="Pagu", x=per_kat_tabel["KATEGORI"], y=per_kat_tabel["Pagu"]),
            go.Bar(name="Realisasi", x=per_kat_tabel["KATEGORI"], y=per_kat_tabel["Realisasi"]),
        ])
        fig_kat.update_layout(barmode="group", yaxis_title="Rupiah", xaxis_title=None)
        st.plotly_chart(fig_kat, use_container_width=True)

        st.dataframe(
            per_kat_tabel.style.format({
                "Pagu": "Rp {:,.0f}", "Realisasi": "Rp {:,.0f}", "Persen Realisasi": "{:.1f}%",
            }),
            use_container_width=True, hide_index=True,
        )
        st.divider()

    # --- Tabel rincian kegiatan/output/suboutput ---
    st.markdown(f"**Rincian {label_kategori}**")
    kolom_tampil = [
        "NMDEPT", "NMSATKER", "KABKOTA", "KEGIATAN", "OUTPUT", "SUBOUTPUT", "AKUN",
        "PAGU", "REALISASI",
    ]
    if kategori is None:
        kolom_tampil = ["KATEGORI"] + kolom_tampil
    rincian = df_scope[kolom_tampil].copy()
    rincian["Persen Realisasi"] = (
        rincian["REALISASI"] / rincian["PAGU"].replace(0, np.nan) * 100
    ).fillna(0)
    rincian = rincian.sort_values("PAGU", ascending=False)

    st.dataframe(
        rincian.style.format({
            "PAGU": "Rp {:,.0f}", "REALISASI": "Rp {:,.0f}", "Persen Realisasi": "{:.1f}%",
        }),
        use_container_width=True, hide_index=True,
    )
    st.caption(
        "Tabel ini menampilkan rincian tiap baris kegiatan/output/suboutput yang termasuk "
        f"{label_kategori.lower()}, diurutkan dari pagu terbesar."
    )

    st.divider()

    # --- AI: narasi otomatis + chat pencarian tematik ---
    def _ringkasan():
        cakupan = f" (khusus satker Anda)" if scope_kdsatker is not None else ""
        return f"""
Data {label_kategori}{cakupan}:
- Cakupan: {kategori or f'Semua {label_kategori}'}
- Tahun: {tahun}
- Pagu: Rp {pagu_total:,.0f}
- Realisasi: Rp {realisasi_total:,.0f} ({persen_serapan:.1f}% dari pagu)
- Sisa pagu: Rp {sisa_pagu:,.0f}
- Proyeksi realisasi akhir tahun: Rp {proyeksi_akhir_tahun:,.0f} ({persen_proyeksi:.1f}% dari pagu)
""".strip()

    cari_fn = buat_cari_generik(
        df, ["NMDEPT", "NMSATKER", "KABKOTA", "KEGIATAN", "OUTPUT", "SUBOUTPUT", "AKUN", "KATEGORI"],
        scope_kdsatker=scope_kdsatker,
    )

    render_ai_section(
        _ringkasan, cari_fn, page_key=page_key,
        narasi_variasi_key=f"{tahun}_{kategori}_{scope_kdsatker}",
        deskripsi_tool=(
            f"Mencari & menjumlahkan pagu/realisasi {label_kategori} berdasarkan kata kunci "
            "(nama satker/kegiatan/output/suboutput/akun/kategori), dan opsional filter provinsi "
            "atau tahun. WAJIB dipakai untuk pertanyaan yang menyebutkan tema/kegiatan atau "
            "satker tertentu yang BUKAN cakupan yang sedang aktif."
        ),
    )


# --------------------------------------------------------------------------
# Upload dataset bebas (khusus super user) + tanya-AI tentang dataset itu.
# Beda dengan render_ai_section (yang pencariannya sudah tahu skema data anggaran),
# di sini datanya BEBAS (struktur apa saja dari file yang diupload), jadi AI diberi tool
# "analisis_dataset" yang aman: filter pakai sintaks pandas .query() (bukan eval bebas)
# + agregasi (sum/mean/count/dll), bukan eksekusi kode Python sembarangan.
# --------------------------------------------------------------------------

def _analisis_dataset_builder(df_upload: pd.DataFrame):
    FUNGSI_VALID = {"sum", "mean", "count", "min", "max", "median", "std", "nunique"}
    # Beberapa pertanyaan user berbentuk "apa saja X" (mis. "apa saja kegiatan PN di riau?")
    # -- ini bukan agregasi angka, tapi minta DAFTAR nilai unik. Model kadang mencoba
    # agg_fungsi="list" utk kasus begini (bukan fungsi pandas asli) -- ditangani khusus di sini
    # supaya tetap terjawab benar (bukan cuma fallback diam-diam ke "sum" yang gagal di kolom teks).
    FUNGSI_DAFTAR_UNIK = {"list", "unique", "distinct"}

    def analisis_dataset(query_filter: str = None, groupby_kolom: list = None,
                          agg_kolom: str = None, agg_fungsi: str = "sum", limit: int = 20) -> dict:
        # Batasi limit (baik terlalu besar MAUPUN 0/negatif dari model) supaya hasil tool tidak
        # pernah membengkak sampai bikin panggilan Groq berikutnya kena error "Please reduce
        # the length of the messages" -- lihat MAKS_BARIS_HASIL_TOOL.
        try:
            limit = int(limit) if limit else 20
        except (TypeError, ValueError):
            limit = 20
        limit = max(1, min(limit, MAKS_BARIS_HASIL_TOOL))

        d = df_upload
        if query_filter:
            try:
                d = d.query(query_filter)
            except Exception as e:
                return {
                    "error": (
                        f"Filter tidak valid: {e}. Gunakan sintaks pandas query, mis. "
                        "\"kolom_a > 100 and kolom_b == 'X'\". Nama kolom yang mengandung spasi "
                        "harus dibungkus backtick, mis. `nama kolom`."
                    )
                }

        if d.empty:
            return {"ditemukan": False, "pesan": "Tidak ada baris yang cocok dengan filter ini."}

        hasil = {"jumlah_baris_cocok": int(len(d))}

        if agg_kolom and agg_kolom not in d.columns:
            return {"error": f"Kolom '{agg_kolom}' tidak ada di dataset. Kolom tersedia: {list(df_upload.columns)}"}

        # Kasus "daftar nilai unik" (agg_fungsi list/unique/distinct) -- utk pertanyaan
        # "apa saja ...", BUKAN agregasi angka.
        if agg_kolom and agg_fungsi in FUNGSI_DAFTAR_UNIK:
            kolom_grup = (groupby_kolom or []) + [agg_kolom]
            kolom_grup = [c for c in dict.fromkeys(kolom_grup) if c in d.columns]  # unik & valid
            unik = d[kolom_grup].drop_duplicates().head(limit)
            hasil["daftar_nilai_unik"] = _potong_teks_dalam_records(unik.to_dict(orient="records"))
            hasil["jumlah_nilai_unik_ditampilkan"] = len(unik)
            hasil["catatan"] = f"Dibatasi maks {limit} baris unik pertama (dari {d[agg_kolom].nunique():,} total nilai unik)."
            return hasil

        fn = agg_fungsi if agg_fungsi in FUNGSI_VALID else "sum"
        try:
            if groupby_kolom and agg_kolom:
                agg_df = d.groupby(groupby_kolom)[agg_kolom].agg(fn).reset_index()
                agg_df = agg_df.sort_values(agg_kolom, ascending=False).head(limit)
                hasil["hasil_agregasi"] = _potong_teks_dalam_records(agg_df.to_dict(orient="records"))
            elif agg_kolom:
                nilai = getattr(d[agg_kolom], fn)()
                try:
                    hasil["hasil"] = float(nilai)
                except (TypeError, ValueError):
                    hasil["hasil"] = str(nilai)
            elif groupby_kolom:
                # groupby tanpa agg_kolom -> anggap user cuma mau daftar kombinasi unik.
                kolom_valid = [c for c in groupby_kolom if c in d.columns]
                unik = d[kolom_valid].drop_duplicates().head(limit) if kolom_valid else d.head(limit)
                hasil["contoh_baris"] = _potong_teks_dalam_records(unik.to_dict(orient="records"))
            else:
                hasil["contoh_baris"] = _potong_teks_dalam_records(d.head(limit).to_dict(orient="records"))
        except Exception as e:
            hasil["error_agregasi"] = str(e)

        return hasil

    return analisis_dataset


def _baca_file_upload(uploaded_file):
    """Baca file upload (CSV/XLSX/XLS) dengan beberapa lapis fallback, & pesan error yang
    jelas kalau gagal. Return (df, catatan) -- catatan berisi peringatan (str) atau None."""
    nama = uploaded_file.name.lower()

    if nama.endswith((".xlsx", ".xls")):
        engine = "openpyxl" if nama.endswith(".xlsx") else None
        try:
            return pd.read_excel(uploaded_file, engine=engine), None
        except ImportError as e:
            paket = "openpyxl" if nama.endswith(".xlsx") else "xlrd"
            raise RuntimeError(
                f"Server belum punya paket Python `{paket}` yang dibutuhkan untuk membaca file "
                f"Excel (.{nama.rsplit('.', 1)[-1]}). Tambahkan baris `{paket}` ke file "
                "`requirements.txt` di repo GitHub aplikasi ini, lalu deploy ulang. "
                f"(Detail teknis: {e})"
            ) from e

    # CSV -- coba beberapa strategi berurutan, dari yang paling standar ke yang paling toleran,
    # supaya lebih tahan terhadap file yang delimiter-nya bukan koma atau ada baris yang
    # jumlah kolomnya tidak konsisten (mis. koma di dalam nilai yang tidak diapit kutip).
    percobaan = [
        dict(),  # default: delimiter koma, engine C (paling cepat, cukup utk CSV standar)
        dict(sep=None, engine="python"),  # auto-deteksi delimiter (titik koma/tab/dll)
        dict(sep=None, engine="python", on_bad_lines="skip"),  # buang baris yang formatnya rusak
    ]
    error_terakhir = None
    for kwargs in percobaan:
        try:
            uploaded_file.seek(0)
            df = pd.read_csv(uploaded_file, **kwargs)
            catatan = None
            if kwargs.get("on_bad_lines") == "skip":
                catatan = (
                    "⚠️ Sebagian baris di file CSV ini jumlah kolomnya tidak konsisten "
                    "(kemungkinan ada delimiter di dalam salah satu nilai yang tidak diapit "
                    "tanda kutip), sehingga baris-baris itu dilewati saat membaca. Data di "
                    "bawah ini kemungkinan tidak 100% lengkap -- sebaiknya periksa ulang file "
                    "sumbernya kalau perlu semua baris ikut terbaca."
                )
            return df, catatan
        except Exception as e:
            error_terakhir = e
            continue
    raise RuntimeError(
        "Gagal membaca file CSV ini walau sudah dicoba beberapa cara (delimiter default, "
        "delimiter otomatis). Kemungkinan formatnya tidak konsisten (jumlah kolom berbeda-beda "
        f"di beberapa baris). Detail teknis: {error_terakhir}"
    ) from error_terakhir


def render_dataset_upload_qa(page_key: str = "upload_dataset"):
    """Fitur upload dataset (CSV/XLSX) bebas + tanya-AI tentang isinya. HANYA dipanggil
    untuk super user (lihat pemanggilannya di view_dashboard_satker.py)."""
    st.subheader("📁 Upload Dataset Tambahan (Super User)")
    st.caption(
        "Upload file CSV atau Excel apa saja, lalu tanya AI tentang isinya di chat box "
        "di bawah. Fitur ini terpisah dari data anggaran utama."
    )

    uploaded_file = st.file_uploader(
        "Upload file CSV atau Excel", type=["csv", "xlsx", "xls"], key=f"uploader_{page_key}"
    )

    if uploaded_file is None:
        st.caption("Belum ada file yang diupload.")
        return

    try:
        df_upload, catatan_baca = _baca_file_upload(uploaded_file)
    except Exception as e:
        st.error(f"Gagal membaca file: {e}")
        return

    if df_upload.empty:
        st.warning("File berhasil dibaca tapi tidak ada baris data.")
        return

    if catatan_baca:
        st.warning(catatan_baca)

    st.success(f"Berhasil memuat **{len(df_upload):,} baris** x **{len(df_upload.columns)} kolom**.")
    with st.expander("Pratinjau data (20 baris pertama)", expanded=False):
        st.dataframe(df_upload.head(20), use_container_width=True)
        st.caption("Tipe data per kolom: " + ", ".join(f"{c} ({df_upload[c].dtype})" for c in df_upload.columns))

    def _ringkasan():
        kolom_info = "\n".join(f"- {c} ({df_upload[c].dtype})" for c in df_upload.columns)
        contoh = df_upload.head(5).to_dict(orient="records")
        return f"""
Dataset yang diupload user ("{uploaded_file.name}"):
- Jumlah baris: {len(df_upload):,}
- Jumlah kolom: {len(df_upload.columns)}
- Daftar kolom & tipe data:
{kolom_info}
- Contoh 5 baris pertama (JSON): {contoh}
""".strip()

    analisis_fn = _analisis_dataset_builder(df_upload)

    # render_ai_section dibuat generik utk tool "cari_anggaran" (dataset anggaran); di sini
    # kita pakai instance terpisah dengan tool "analisis_dataset" -- jadi bagian narasi/chat
    # ditulis ulang secara ringkas khusus utk dataset upload (skema tool beda).
    _render_chat_dataset_upload(_ringkasan, analisis_fn, page_key, list(df_upload.columns))


def _render_chat_dataset_upload(ringkasan_fn, analisis_fn, page_key: str, daftar_kolom: list):
    import json as _json

    client = get_groq_client()

    tools_groq = [{
        "type": "function",
        "function": {
            "name": "analisis_dataset",
            "description": (
                "Memfilter, mengelompokkan, dan menghitung agregasi (sum/mean/count/min/max/dll) "
                "pada dataset yang diupload user. WAJIB dipakai untuk menjawab pertanyaan spesifik "
                "tentang angka/jumlah/rata-rata di data ini -- jangan mengarang dari ingatan."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query_filter": {
                        "type": ["string", "null"],
                        "description": (
                            "Filter dengan sintaks pandas query (opsional, boleh null), mis. "
                            "\"umur > 30 and kota == 'Jakarta'\". Kosongkan/null utk semua baris. "
                            f"Kolom tersedia: {daftar_kolom}"
                        ),
                    },
                    "groupby_kolom": {
                        "type": ["array", "null"], "items": {"type": "string"},
                        "description": "Kolom utk dikelompokkan (opsional, boleh null).",
                    },
                    "agg_kolom": {
                        "type": ["string", "null"],
                        "description": "Kolom yang mau dihitung/diagregasi (opsional, boleh null).",
                    },
                    "agg_fungsi": {
                        "type": ["string", "null"],
                        "enum": [
                            "sum", "mean", "count", "min", "max", "median", "std", "nunique",
                            "list", "unique", "distinct", None,
                        ],
                        "description": (
                            "Fungsi agregasi, default 'sum'. Utk pertanyaan 'apa saja ...' (minta "
                            "DAFTAR nilai, bukan angka), pakai 'list'/'unique'/'distinct' dgn "
                            "agg_kolom = kolom yang mau didaftar isinya."
                        ),
                    },
                    "limit": {
                        "type": ["integer", "null"],
                        "description": f"Batas baris hasil ditampilkan, default 20, maksimal {MAKS_BARIS_HASIL_TOOL}.",
                    },
                },
                "required": [],
            },
        },
    }]

    def _jalankan_tool_call(nama_fungsi, args):
        if nama_fungsi == "analisis_dataset":
            hasil = analisis_fn(
                query_filter=args.get("query_filter"),
                groupby_kolom=args.get("groupby_kolom"),
                agg_kolom=args.get("agg_kolom"),
                agg_fungsi=args.get("agg_fungsi") or "sum",
                limit=args.get("limit") or 20,
            )
        else:
            hasil = {"error": f"Fungsi tidak dikenal: {nama_fungsi}"}
        return _json_tool_aman(hasil)

    if client is None:
        st.info(
            "Chat AI belum aktif. Tambahkan `GROQ_API_KEY` di file `.streamlit/secrets.toml` "
            "(lihat README.md) untuk mengaktifkan fitur ini."
        )
        return

    chat_state_key = f"chat_history_{page_key}"
    if chat_state_key not in st.session_state:
        st.session_state[chat_state_key] = []

    MAKS_HISTORI_DIKIRIM = 6

    col_title, col_reset = st.columns([5, 1])
    with col_title:
        st.markdown("**💬 Tanya AI tentang Dataset Ini**")
    with col_reset:
        if st.button("🗑️ Reset Chat", key=f"resetchat_{page_key}"):
            st.session_state[chat_state_key] = []
            st.rerun()

    for msg in st.session_state[chat_state_key]:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])

    prompt = st.chat_input("Tulis pertanyaan tentang dataset yang diupload...", key=f"chatinput_{page_key}")

    if prompt:
        st.session_state[chat_state_key].append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.write(prompt)

        with st.spinner("AI sedang menjawab..."):
            system_msg = {
                "role": "system",
                "content": (
                    "Kamu adalah asisten analisis data. Jawab pertanyaan user tentang dataset yang "
                    "diupload memakai tool analisis_dataset -- jangan mengarang angka. Kalau filter "
                    "gagal atau kolom tidak ada, jelaskan errornya ke user dengan jujur dan sarankan "
                    "perbaikan (mis. nama kolom yang benar).\n\n" + ringkasan_fn()
                ),
            }
            histori_dikirim = st.session_state[chat_state_key][-MAKS_HISTORI_DIKIRIM:]
            messages = [system_msg] + histori_dikirim

            jawaban = None
            try:
                resp = _groq_chat_dgn_fallback(
                    client, messages=messages, tools=tools_groq, tool_choice="auto",
                )
                msg = resp.choices[0].message

                if msg.tool_calls:
                    messages.append({
                        "role": "assistant",
                        "content": msg.content or "",
                        "tool_calls": [
                            {
                                "id": tc.id, "type": "function",
                                "function": {"name": tc.function.name, "arguments": tc.function.arguments},
                            }
                            for tc in msg.tool_calls
                        ],
                    })
                    for tc in msg.tool_calls:
                        try:
                            args = _json.loads(tc.function.arguments)
                        except Exception:
                            args = {}
                        hasil_tool = _jalankan_tool_call(tc.function.name, args)
                        messages.append({"role": "tool", "tool_call_id": tc.id, "content": hasil_tool})
                    resp2 = _groq_chat_dgn_fallback(client, messages=messages)
                    jawaban = resp2.choices[0].message.content
                else:
                    jawaban = msg.content
            except Exception as e:
                pesan_lower = _pesan_error_groq(e)
                detail = getattr(e, "message", None) or str(e)
                if "reduce the length" in pesan_lower or ("context" in pesan_lower and "length" in pesan_lower):
                    jawaban = (
                        "⚠️ Percakapan atau hasil query-nya terlalu besar buat diproses AI. Coba: "
                        "(1) klik **🗑️ Reset Chat** untuk mulai percakapan baru, atau (2) ajukan "
                        "pertanyaan yang lebih spesifik/sempit (mis. sebutkan nilai/filter yang "
                        "lebih pasti supaya hasilnya tidak terlalu banyak baris)."
                    )
                elif "tool_use_failed" in pesan_lower or "tool choice" in pesan_lower:
                    jawaban = (
                        "⚠️ Model AI sedang tidak stabil untuk fitur analisis data ini (sudah "
                        "dicoba dgn model cadangan, tetap gagal). Coba tanya ulang beberapa saat lagi."
                    )
                else:
                    jawaban = f"⚠️ Gagal memanggil Groq API: {detail}"

        st.session_state[chat_state_key].append({"role": "assistant", "content": jawaban})
        with st.chat_message("assistant"):
            st.write(jawaban)


# --------------------------------------------------------------------------
# Confidence Score, Early Warning, Heatmap Deviasi & Waterfall -- FITUR BARU,
# dibangun di atas mesin granular yang sama (fungsi di atas). Tidak menyentuh
# fungsi lain -- aman ditambahkan tanpa mengubah perilaku halaman yang sudah ada.
# --------------------------------------------------------------------------

def hitung_detail_granular(df_all: pd.DataFrame, tahun_y: int, filter_dict: dict) -> pd.DataFrame:
    """Detail hasil forecast per (KDSATKER, JENIS BELANJA, AKUN) utk cakupan filter_dict --
    dipakai utk visual tambahan (confidence score, early warning, heatmap, waterfall).
    Return DataFrame kosong kalau tidak ada data. Setiap baris = satu kombinasi granular,
    dgn kolom PAGU, REALISASI_SD_INI, BASELINE_FORECAST_TOTAL, RATE_HIST, CV_RATE,
    N_TAHUN, SUMBER_PROFIL, SKOR_KEPERCAYAAN.
    """
    df_all = _agregasi_ke_kunci_granular(df_all)
    bobot_tahun = _bobot_per_tahun_absolut(tahun_y, BOBOT_TAHUN)
    df_now = _filter_entitas(df_all, tahun_y, filter_dict)
    if df_now.empty:
        return pd.DataFrame()
    df_hist = df_all[df_all["TAHUN"].isin(bobot_tahun.keys())]
    for kolom, nilai in filter_dict.items():
        if nilai is not None:
            df_hist = df_hist[df_hist[kolom] == nilai]
    if df_hist.empty:
        return pd.DataFrame()

    gabung = _gabungkan_fallback_granular(df_now, df_hist, bobot_tahun, BULAN_KOLOM)
    baseline = _baseline_forecast_granular(gabung, BULAN_KOLOM)

    _, bulan_penuh_terakhir = hitung_bulan_penuh_terakhir(df_now, tahun_y)
    realisasi_sd_ini = df_now[BULAN_KOLOM[:bulan_penuh_terakhir]].sum(axis=1) if bulan_penuh_terakhir else pd.Series(0.0, index=df_now.index)

    hasil = df_now[["KDSATKER", "NMSATKER", "KDDEPT", "NMDEPT", "JENIS BELANJA", "LABEL_JENIS_BELANJA", "AKUN", "PAGU"]].copy()
    hasil["REALISASI_SD_INI"] = realisasi_sd_ini.values
    hasil["BASELINE_FORECAST_TOTAL"] = baseline.sum(axis=1)
    hasil["RATE_HIST"] = gabung["RATE_HIST"].values
    hasil["CV_RATE"] = gabung["CV_RATE"].values
    hasil["N_TAHUN"] = gabung["N_TAHUN"].values
    hasil["SUMBER_PROFIL"] = gabung["SUMBER_PROFIL"].values

    # Target akhir setelah rekonsiliasi sederhana (setara logika isi_tabel_proyeksi
    # per baris granular): tidak boleh < realisasi yg sudah terjadi; utk kategori
    # selain Pegawai, dibatasi maksimal pagu.
    is_pegawai = hasil["JENIS BELANJA"] == 51
    target = np.maximum(hasil["BASELINE_FORECAST_TOTAL"], hasil["REALISASI_SD_INI"])
    target = np.where(is_pegawai, target, np.minimum(target, hasil["PAGU"]))
    hasil["FORECAST_TOTAL"] = target

    faktor_sumber = hasil["SUMBER_PROFIL"].map({
        "Satker sendiri": 1.00, "Kelompok sejenis (K/L+Jenis+Pagu)": 0.70,
        "Rata-rata nasional per Jenis Belanja": 0.50, "Tidak ada histori sama sekali": 0.20,
    }).fillna(0.5)
    faktor_n = np.clip(hasil["N_TAHUN"] / 3.0, 0.5, 1.0)
    hasil["SKOR_KEPERCAYAAN"] = (100 * (1 / (1 + hasil["CV_RATE"].clip(lower=0))) * faktor_sumber * faktor_n).clip(0, 100).round(1)

    hasil["_PROFIL_BULANAN"] = list(gabung[BULAN_KOLOM].to_numpy())  # dipakai heatmap
    return hasil


def deteksi_early_warning(df_all: pd.DataFrame, tahun_y: int, filter_dict: dict,
                           ambang_deviasi_pp: float = 15.0, ambang_z: float = 1.5) -> pd.DataFrame:
    """Early warning per kombinasi granular: forecast melebihi pagu, penyerapan
    terlalu cepat/lambat dibanding pola historis bulan yang sama, lonjakan tak wajar
    (z-score bulan terakhir vs histori), + skor risiko (0-100) & alasan.
    """
    detail = hitung_detail_granular(df_all, tahun_y, filter_dict)
    if detail.empty:
        return detail

    _, bulan_penuh_terakhir = hitung_bulan_penuh_terakhir(_filter_entitas(df_all, tahun_y, filter_dict), tahun_y)
    detail["RASIO_FORECAST_PAGU"] = np.where(detail["PAGU"] > 0, detail["FORECAST_TOTAL"] / detail["PAGU"], np.nan)
    detail["FLAG_OVER_PAGU"] = detail["RASIO_FORECAST_PAGU"] > 1.001
    detail["OVER_PAGU_INFO_SAJA"] = detail["FLAG_OVER_PAGU"] & (detail["JENIS BELANJA"] == 51)

    df_agg = _agregasi_ke_kunci_granular(df_all)
    bobot_tahun = _bobot_per_tahun_absolut(tahun_y, BOBOT_TAHUN)
    df_hist = df_agg[df_agg["TAHUN"].isin(bobot_tahun.keys())]
    for kolom, nilai in filter_dict.items():
        if nilai is not None:
            df_hist = df_hist[df_hist[kolom] == nilai]

    key_cols = KUNCI_GRANULAR
    if bulan_penuh_terakhir > 0:
        dh = df_hist.copy()
        dh["_KUM"] = dh[BULAN_KOLOM[:bulan_penuh_terakhir]].sum(axis=1)
        dh["_PCT"] = np.where(dh["PAGU"] > 0, dh["_KUM"] / dh["PAGU"] * 100, np.nan)
        dh["_BOBOT"] = dh["TAHUN"].map(bobot_tahun).fillna(0.0)
        dhv = dh.dropna(subset=["_PCT"])
        dhv = dhv[dhv["_BOBOT"] > 0].assign(_WPCT=lambda x: x["_PCT"] * x["_BOBOT"])
        agg = dhv.groupby(key_cols).agg(_WPCT_SUM=("_WPCT", "sum"), _BOBOT_SUM=("_BOBOT", "sum"))
        pct_hist = (agg["_WPCT_SUM"] / agg["_BOBOT_SUM"]).rename("PCT_KUM_HIST")
        detail = detail.merge(pct_hist.reset_index(), on=key_cols, how="left")
        detail["PCT_KUM_AKTUAL"] = np.where(detail["PAGU"] > 0, detail["REALISASI_SD_INI"] / detail["PAGU"] * 100, np.nan)
        detail["DEVIASI_KECEPATAN_PP"] = (detail["PCT_KUM_AKTUAL"] - detail["PCT_KUM_HIST"]).fillna(0.0)

        bulan_terakhir_kol = BULAN_KOLOM[bulan_penuh_terakhir - 1]
        stat = df_hist.groupby(key_cols)[bulan_terakhir_kol].agg(_MEAN="mean", _STD="std").fillna(0)
        detail = detail.merge(stat.reset_index(), on=key_cols, how="left")
        df_now_bulan = _filter_entitas(df_all, tahun_y, filter_dict)
        df_now_bulan = _agregasi_ke_kunci_granular(df_now_bulan)[key_cols + [bulan_terakhir_kol]]
        detail = detail.merge(df_now_bulan, on=key_cols, how="left")
        std_aman = detail["_STD"].replace(0, np.nan)
        detail["Z_LONJAKAN"] = ((detail[bulan_terakhir_kol] - detail["_MEAN"]) / std_aman).fillna(0)
        detail = detail.drop(columns=["_MEAN", "_STD", bulan_terakhir_kol])
    else:
        detail["DEVIASI_KECEPATAN_PP"] = 0.0
        detail["Z_LONJAKAN"] = 0.0

    detail["FLAG_TERLALU_CEPAT"] = detail["DEVIASI_KECEPATAN_PP"] > ambang_deviasi_pp
    detail["FLAG_TERLALU_LAMBAT"] = detail["DEVIASI_KECEPATAN_PP"] < -ambang_deviasi_pp
    detail["FLAG_LONJAKAN"] = detail["Z_LONJAKAN"] > ambang_z

    komponen_over = np.clip((detail["RASIO_FORECAST_PAGU"].fillna(1) - 1) * 100, 0, 40) * (~detail["OVER_PAGU_INFO_SAJA"])
    komponen_kecepatan = np.clip(detail["DEVIASI_KECEPATAN_PP"].abs() / 50 * 30, 0, 30)
    komponen_lonjakan = np.clip(detail["Z_LONJAKAN"] / 4 * 30, 0, 30)
    detail["SKOR_RISIKO"] = (komponen_over + komponen_kecepatan + komponen_lonjakan).clip(0, 100).round(1)

    def _alasan(row):
        a = []
        if row["FLAG_OVER_PAGU"]:
            a.append(("Forecast melebihi pagu (wajar, Belanja Pegawai)" if row["OVER_PAGU_INFO_SAJA"]
                      else "Forecast melebihi pagu") + f" ({row['RASIO_FORECAST_PAGU']*100:.0f}% dari pagu)")
        if row["FLAG_TERLALU_CEPAT"]:
            a.append(f"Penyerapan lebih cepat {row['DEVIASI_KECEPATAN_PP']:.1f} pp dari pola historis bulan yang sama")
        if row["FLAG_TERLALU_LAMBAT"]:
            a.append(f"Penyerapan lebih lambat {abs(row['DEVIASI_KECEPATAN_PP']):.1f} pp dari pola historis bulan yang sama")
        if row["FLAG_LONJAKAN"]:
            a.append(f"Lonjakan realisasi tidak wajar bulan terakhir (z-score {row['Z_LONJAKAN']:.1f})")
        return "; ".join(a) if a else "Tidak ada anomali terdeteksi"

    detail["ALASAN_RISIKO"] = detail.apply(_alasan, axis=1)
    return detail.drop(columns=["_PROFIL_BULANAN"], errors="ignore")


def hitung_deviasi_heatmap(df_all: pd.DataFrame, tahun_y: int, filter_dict: dict) -> pd.DataFrame:
    """Matrix deviasi realisasi aktual vs ekspektasi profil historis (%), per satker x
    bulan, hanya utk bulan yang sudah aktual. Positif = lebih tinggi dari biasanya;
    negatif = lebih rendah/tertinggal dari pola normal."""
    detail = hitung_detail_granular(df_all, tahun_y, filter_dict)
    if detail.empty:
        return pd.DataFrame()
    _, bulan_penuh_terakhir = hitung_bulan_penuh_terakhir(_filter_entitas(df_all, tahun_y, filter_dict), tahun_y)
    if bulan_penuh_terakhir == 0:
        return pd.DataFrame()

    df_agg = _agregasi_ke_kunci_granular(df_all)
    bobot_tahun = _bobot_per_tahun_absolut(tahun_y, BOBOT_TAHUN)
    df_now = _filter_entitas(df_agg, tahun_y, filter_dict)
    df_hist = df_agg[df_agg["TAHUN"].isin(bobot_tahun.keys())]
    for kolom, nilai in filter_dict.items():
        if nilai is not None:
            df_hist = df_hist[df_hist[kolom] == nilai]
    gabung = _gabungkan_fallback_granular(df_now, df_hist, bobot_tahun, BULAN_KOLOM)
    gabung = gabung.merge(df_now[KUNCI_GRANULAR + ["NMSATKER", "PAGU"]], on=KUNCI_GRANULAR, how="left", suffixes=("", "_asli"))

    pagu_grp = gabung.groupby("NMSATKER")["PAGU"].sum().replace(0, np.nan)
    hasil_bulanan = {}
    df_now_idx = df_now.set_index(KUNCI_GRANULAR)
    for b in BULAN_KOLOM[:bulan_penuh_terakhir]:
        aktual = df_now.groupby("NMSATKER")[b].sum()
        ekspektasi = (gabung[b] * gabung["PAGU"]).groupby(gabung["NMSATKER"]).sum()
        hasil_bulanan[BULAN_LABEL[BULAN_KOLOM.index(b) + 1]] = (aktual - ekspektasi) / pagu_grp * 100
    return pd.DataFrame(hasil_bulanan)


def ringkas_waterfall(df_all: pd.DataFrame, tahun_y: int, filter_dict: dict) -> dict:
    """Ringkasan angka utk grafik waterfall Realisasi -> Forecast -> Total (vs Pagu)."""
    detail = hitung_detail_granular(df_all, tahun_y, filter_dict)
    if detail.empty:
        return {"pagu": 0, "realisasi_sd_ini": 0, "forecast_sisa_tahun": 0, "forecast_total": 0,
                "sisa_tak_terserap": 0, "kelebihan_dari_pagu": 0}
    pagu = float(detail["PAGU"].sum())
    realisasi = float(detail["REALISASI_SD_INI"].sum())
    forecast_total = float(detail["FORECAST_TOTAL"].sum())
    forecast_sisa = max(forecast_total - realisasi, 0.0)
    return {
        "pagu": pagu, "realisasi_sd_ini": realisasi, "forecast_sisa_tahun": forecast_sisa,
        "forecast_total": forecast_total,
        "sisa_tak_terserap": max(pagu - forecast_total, 0.0),
        "kelebihan_dari_pagu": max(forecast_total - pagu, 0.0),
    }
