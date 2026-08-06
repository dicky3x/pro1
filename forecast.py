"""
forecast.py — Mesin hybrid forecasting realisasi belanja APBN.
===============================================================
Pengganti logika proyeksi lama (common.py::hitung_proyeksi_agregat,
hitung_proyeksi_per_kategori, isi_tabel_proyeksi, sesuaikan_proyeksi_tukin_kemenhan).

METODOLOGI
----------
1.  Monthly spending profile per kombinasi SATKER × JENIS BELANJA (2 digit) ×
    AKUN (6 digit), dibentuk dari weighted average pola historis 5 tahun:
        y-1 = 50%, y-2 = 25%, y-3 = 12,5%, y-4 = 6,25%, y-5 = 6,25%
    (untuk tahun target 2026 ini persis 2025=50% … 2021=6,25%).
2.  ROLLING FORECAST: bulan yang sudah penuh (<= bulan_penuh_terakhir) dipegang
    sebagai AKTUAL; sisa target akhir tahun didistribusikan ke bulan-bulan
    berikutnya mengikuti ekor profil historis yang DINORMALISASI:
        forecast_m = (Target − Realisasi_sd_bulan_k) × P_m / Σ_{j>k} P_j
    Setiap kali realisasi bulan baru masuk (k bertambah), seluruh forecast
    bulan berikutnya otomatis dihitung ulang.
3.  SATKER BARU tanpa histori → profil satker SEJENIS (K/L × jenis belanja ×
    kelompok pagu) → profil K/L → profil jenis belanja nasional → rata-rata
    semua satker (berjenjang, tercatat di kolom METODE).
4.  BELANJA PEGAWAI (51): BOLEH melebihi pagu (target = rata-rata tertimbang
    RUPIAH historis, bukan rate × pagu) dan mendukung POLICY ADJUSTMENT lewat
    tabel konfigurasi eksternal (contoh: indeks tukin Kemenhan 70% → 90%
    mulai Juli, rapel September) tanpa mengubah kode program.
5.  BELANJA BARANG (52) & BELANJA MODAL (53): target = tingkat serapan
    historis × pagu, dibatasi maksimal pagu, pola bulanan mengikuti histori.
6.  OUTPUT TAMBAHAN: confidence score (dari koefisien variasi pola historis
    antar tahun), early warning (forecast > pagu, serapan terlalu
    cepat/lambat, lonjakan tidak wajar, skor risiko + alasan), dan helper
    visualisasi plotly: aktual vs forecast, forecast bulanan, heatmap
    deviasi antar satker, waterfall pagu–realisasi–forecast.

PERFORMA: seluruh perhitungan vektorisasi pandas/numpy (groupby + operasi
matriks), tanpa loop per satker; agregasi dilakukan sekali di level kunci
terkecil. Aman untuk ratusan ribu baris.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from datetime import date

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Konstanta & konfigurasi
# ---------------------------------------------------------------------------
BULAN = ["JAN", "FEB", "MAR", "APR", "MEI", "JUN",
         "JUL", "AGS", "SEP", "OKT", "NOV", "DES"]

# Bobot tahun relatif terhadap tahun target Y: tahun Y-i berbobot BOBOT_TAHUN[i].
# Untuk target 2026 → 2025=50%, 2024=25%, 2023=12,5%, 2022=6,25%, 2021=6,25%.
BOBOT_TAHUN = {1: 0.50, 2: 0.25, 3: 0.125, 4: 0.0625, 5: 0.0625}
TOTAL_BOBOT = sum(BOBOT_TAHUN.values())  # = 1.0

JB_PEGAWAI, JB_BARANG, JB_MODAL = 51, 52, 53

# Kelompok pagu (per satker per tahun) utk fallback "satker sejenis".
GRUP_PAGU_BATAS = [0, 5e8, 2e9, 1e10, np.inf]
GRUP_PAGU_LABEL = ["<500jt", "500jt–2M", "2–10M", ">10M"]

# Ambang early warning (boleh di-override lewat argumen deteksi_peringatan).
AMBANG_SERAPAN_CEPAT = 1.25   # aktual > 125% ekspektasi historis → terlalu cepat
AMBANG_SERAPAN_LAMBAT = 0.75  # aktual < 75% ekspektasi historis  → terlalu lambat
AMBANG_LONJAKAN_RASIO = 2.5   # realisasi bulan terakhir > 2,5× ekspektasi bulanan

KUNCI_DETAIL = ["KDSATKER", "JENIS BELANJA", "AKUN"]

# Konfigurasi default policy adjustment Belanja Pegawai (bisa diganti file CSV,
# lihat muat_kebijakan()). Contoh: kenaikan indeks tukin Kemenhan 70% → 90%.
KEBIJAKAN_DEFAULT = pd.DataFrame([{
    "tahun": 2026,
    "kddept": 12,                 # Kementerian Pertahanan
    "akun": "512411",             # beberapa akun boleh dipisah ";"
    "mulai_bulan": 7,             # indeks baru berlaku mulai Juli
    "rapel_bulan": 9,             # rapel Juli–Agustus dibayar September
    "indeks_lama": 0.70,
    "indeks_baru": 0.90,
    "keterangan": "Kenaikan indeks tunjangan kinerja Kemenhan 70% → 90%",
}])


def muat_kebijakan(path: str = "config_kebijakan_pegawai.csv") -> pd.DataFrame:
    """Muat tabel konfigurasi policy adjustment Belanja Pegawai.

    Kalau file CSV ada → dipakai (sehingga perubahan kebijakan TIDAK perlu
    mengubah kode program). Kalau tidak ada → pakai KEBIJAKAN_DEFAULT.
    Skema kolom: tahun, kddept, akun (kode 6 digit; multi dipisah ';'),
    mulai_bulan, rapel_bulan (opsional), indeks_lama, indeks_baru, keterangan.
    """
    if path and os.path.exists(path):
        kb = pd.read_csv(path, dtype={"akun": str})
        return kb
    return KEBIJAKAN_DEFAULT.copy()


# ---------------------------------------------------------------------------
# Normalisasi sumber data (Excel/CSV dengan berbagai variasi nama kolom)
# ---------------------------------------------------------------------------
_PETA_KOLOM = {
    "tahun": "TAHUN",
    "kementerian_kode": "KDDEPT", "kode kementerian": "KDDEPT",
    "kementerian_uraian": "NMDEPT", "kementerian/lembaga": "NMDEPT",
    "satker_kode": "KDSATKER", "kode satker": "KDSATKER",
    "satker_uraian": "NMSATKER", "nama satker": "NMSATKER",
    "provinsi_kode": "KDPROV", "provinsi_uraian": "PROVINSI", "provinsi": "PROVINSI",
    "kabkota_kode": "KDKABKOTA", "kabkota_uraian": "KABKOTA",
    "jenis belanja": "JENIS BELANJA", "jenis_belanja": "JENIS BELANJA",
    "akun_kode": "AKUN", "akun": "AKUN",
    "akun_uraian": "NM_AKUN",
    "pagu_dipa": "PAGU", "pagu": "PAGU",
    "blokir": "BLOKIR",
    "jan": "JAN", "januari": "JAN", "feb": "FEB", "februari": "FEB",
    "mar": "MAR", "maret": "MAR", "apr": "APR", "april": "APR",
    "mei": "MEI", "jun": "JUN", "juni": "JUN",
    "jul": "JUL", "juli": "JUL", "ags": "AGS", "agu": "AGS", "agustus": "AGS",
    "sep": "SEP", "sept": "SEP", "september": "SEP",
    "okt": "OKT", "oktober": "OKT", "nov": "NOV", "november": "NOV",
    "des": "DES", "desember": "DES",
}


def normalisasi_sumber(df_mentah: pd.DataFrame) -> pd.DataFrame:
    """Seragamkan kolom sumber (format lebar jan..des ATAU format panjang
    dengan kolom 'Bulan' + 'Realisasi') menjadi skema internal:
    TAHUN, KDDEPT, NMDEPT, KDSATKER, NMSATKER, PROVINSI, JENIS BELANJA,
    AKUN, PAGU, BLOKIR, JAN..DES, REALISASI, SISA PAGU.
    """
    d = df_mentah.copy()
    d.columns = [str(c).strip().lower() for c in d.columns]
    d = d.rename(columns={c: _PETA_KOLOM[c] for c in d.columns if c in _PETA_KOLOM})

    # Format panjang (ada kolom BULAN & REALISASI) → pivot ke lebar.
    if "BULAN" in d.columns and "REALISASI" in d.columns:
        d["BULAN"] = d["BULAN"].astype(str).str.strip().str.lower().str[:3]
        index = [c for c in d.columns if c not in ("BULAN", "REALISASI")]
        d = (d.pivot_table(index=index, columns="BULAN", values="REALISASI",
                           aggfunc="sum", fill_value=0)
               .reset_index().rename(columns=_PETA_KOLOM))

    for b in BULAN:
        if b not in d.columns:
            d[b] = 0.0
        d[b] = pd.to_numeric(d[b], errors="coerce").fillna(0.0)

    d["TAHUN"] = d["TAHUN"].astype(int)
    d["PAGU"] = pd.to_numeric(d.get("PAGU", 0), errors="coerce").fillna(0.0)
    if "BLOKIR" not in d.columns:
        d["BLOKIR"] = 0.0
    d["BLOKIR"] = pd.to_numeric(d["BLOKIR"], errors="coerce").fillna(0.0)
    d["KDSATKER"] = d["KDSATKER"].astype(str).str.replace(r"\.0$", "", regex=True)
    d["AKUN"] = d["AKUN"].astype(str).str.replace(r"\.0$", "", regex=True).str.zfill(6)
    d["JENIS BELANJA"] = pd.to_numeric(d["JENIS BELANJA"], errors="coerce")
    # Jenis belanja 2 digit: dari kolom sendiri, atau turunan kode akun.
    d["JENIS BELANJA"] = d["JENIS BELANJA"].fillna(
        d["AKUN"].str[:2]).astype(int)
    if "KDDEPT" in d.columns:
        d["KDDEPT"] = pd.to_numeric(d["KDDEPT"], errors="coerce").astype("Int64")
    # REALISASI selalu dihitung ulang dari kolom bulanan agar konsisten.
    d["REALISASI"] = d[BULAN].sum(axis=1)
    d["SISA PAGU"] = d["PAGU"] - d["REALISASI"]
    return d


def muat_data(path: str) -> pd.DataFrame:
    """Baca CSV/Excel sumber lalu normalisasi (auto-detect delimiter)."""
    lower = path.lower()
    if lower.endswith((".xlsx", ".xls")):
        raw = pd.read_excel(path)
    else:
        raw = pd.read_csv(path, sep=None, engine="python",
                          compression="infer" if lower.endswith(".gz") else None)
    return normalisasi_sumber(raw)


# ---------------------------------------------------------------------------
# Utilitas agregasi & bulan penuh
# ---------------------------------------------------------------------------
def agregasi(df: pd.DataFrame, kunci: list) -> pd.DataFrame:
    """Agregasi ke level `kunci` (sekali panggil, vektorisasi penuh)."""
    agg = df.groupby(kunci, as_index=False).agg(
        PAGU=("PAGU", "sum"), **{b: (b, "sum") for b in BULAN})
    agg["REALISASI"] = agg[BULAN].sum(axis=1)
    return agg


def grup_pagu(pagu: pd.Series) -> pd.Series:
    return pd.cut(pagu.fillna(0), GRUP_PAGU_BATAS,
                  labels=GRUP_PAGU_LABEL, include_lowest=True)


def hitung_bulan_penuh(df_tahun: pd.DataFrame, tahun: int) -> tuple[int, int]:
    """Return (bulan_terakhir_terisi, bulan_penuh_terakhir).

    - Tahun lampau      → semua bulan terisi dianggap penuh.
    - Tahun mendatang  → 0.
    - Tahun berjalan    → bulan terakhir yang terisi diasumsikan BELUM penuh
      (data parsial), jadi bulan penuh terakhir = bulan_terakhir − 1.
    """
    monthly = df_tahun[BULAN].sum().values
    terisi = np.nonzero(monthly)[0] + 1
    bulan_terakhir = int(terisi.max()) if len(terisi) else 0
    hari_ini = date.today()
    if tahun < hari_ini.year:
        return bulan_terakhir, bulan_terakhir
    if tahun > hari_ini.year:
        return bulan_terakhir, 0
    return bulan_terakhir, max(bulan_terakhir - 1, 0)


# ---------------------------------------------------------------------------
# Inti: profil historis tertimbang (dipakai utk level detail & fallback)
# ---------------------------------------------------------------------------
def _profil_tertimbang(df_level: pd.DataFrame, tahun_target: int, kunci: list):
    """Weighted average profil bulanan pada level `kunci`.

    Input : df_level teragregasi per kunci + TAHUN (kolom PAGU, REALISASI, JAN..DES).
    Output: dict berisi
      profil         : share bulanan (Σ per baris = 1); NaN bila tak ada histori
      tingkat        : tingkat serapan tahunan tertimbang (realisasi/pagu)
      rupiah_bulanan : rata-rata tertimbang RUPIAH per bulan (utk belanja 51)
      cv             : koefisien variasi share antar tahun (dasar confidence)
      bobot_tersedia : Σ bobot tahun yg punya data (kelengkapan, maks 1.0)
      n_tahun        : banyaknya tahun kontribusi
    """
    g = df_level.copy()
    g["SELISIH"] = tahun_target - g["TAHUN"]
    g["BOBOT"] = g["SELISIH"].map(BOBOT_TAHUN)
    g = g[g["BOBOT"].notna() & (g["BOBOT"] > 0)]
    if g.empty:
        return None
    for c in kunci:
        g[c] = g[c].astype(str)

    g["RATE"] = np.where(g["PAGU"] > 0, g["REALISASI"] / g["PAGU"], np.nan)

    idx = pd.MultiIndex.from_frame(g[kunci].drop_duplicates())
    S1 = pd.DataFrame(0.0, index=idx, columns=BULAN)   # Σ w·share
    S2 = pd.DataFrame(0.0, index=idx, columns=BULAN)   # Σ w·share²
    RP = pd.DataFrame(0.0, index=idx, columns=BULAN)   # Σ w·rupiah
    Ws = pd.Series(0.0, index=idx)                     # Σ w (share valid)
    Wr = pd.Series(0.0, index=idx)                     # Σ w (baris ada)
    Rn = pd.Series(0.0, index=idx)                     # Σ w·rate
    Rd = pd.Series(0.0, index=idx)                     # Σ w (rate valid)
    Cn = pd.Series(0.0, index=idx)                     # jumlah tahun kontribusi

    for _, sub in g.groupby("TAHUN"):
        w = float(sub["BOBOT"].iloc[0])
        s = sub.set_index(kunci)
        share = s[BULAN].div(s["REALISASI"].replace(0, np.nan), axis=0)
        ada_share = share.notna().any(axis=1).astype(float)
        S1 = S1.add(share.fillna(0.0) * w, fill_value=0.0)
        S2 = S2.add(share.fillna(0.0) ** 2 * w, fill_value=0.0)
        RP = RP.add(s[BULAN].fillna(0.0) * w, fill_value=0.0)
        Wr = Wr.add(pd.Series(w, index=s.index), fill_value=0.0)
        Ws = Ws.add(ada_share * w, fill_value=0.0)
        Rn = Rn.add(s["RATE"].fillna(0.0) * w, fill_value=0.0)
        Rd = Rd.add(s["RATE"].notna().astype(float) * w, fill_value=0.0)
        Cn = Cn.add(ada_share, fill_value=0.0)

    # Rata-rata tertimbang share + koefisien variasi antar tahun.
    mean_share = S1.div(Ws.replace(0, np.nan), axis=0)
    mean_sq = S2.div(Ws.replace(0, np.nan), axis=0)
    var = (mean_sq - mean_share ** 2).clip(lower=0.0)
    cv_bulanan = np.sqrt(var).div(mean_share.replace(0, np.nan), axis=0)

    profil = mean_share.div(mean_share.sum(axis=1).replace(0, np.nan), axis=0)
    profil = profil.where(Ws > 0)                      # tanpa histori → NaN
    cv = (cv_bulanan.fillna(0.0) * profil.fillna(0.0)).sum(axis=1)

    return {
        "profil": profil.fillna(0.0).where(Ws > 0),
        "tingkat": Rn.div(Rd.replace(0, np.nan)).where(Rd > 0),
        "rupiah_bulanan": RP.div(Wr.replace(0, np.nan), axis=0),
        "cv": cv,
        "bobot_tersedia": Ws,
        "n_tahun": Cn,
    }


def _sejajarkan(res, df_target: pd.DataFrame, kunci: list):
    """Sejajarkan hasil _profil_tertimbang ke baris df_target (posisi 1-1)."""
    if res is None:
        return None
    mi = pd.MultiIndex.from_frame(df_target[kunci].astype(str))
    return (res["profil"].reindex(mi),
            res["tingkat"].reindex(mi),
            res["cv"].reindex(mi),
            res["bobot_tersedia"].reindex(mi),
            res["n_tahun"].reindex(mi),
            res["rupiah_bulanan"].reindex(mi))


# ---------------------------------------------------------------------------
# Hasil forecast
# ---------------------------------------------------------------------------
@dataclass
class HasilForecast:
    tahun: int
    bulan_penuh: int                      # bulan penuh terakhir (0..12)
    detail: pd.DataFrame                  # level satker × JB × akun
    per_satker: pd.DataFrame              # agregasi satker (+ kolom EXP_*)
    per_jenis: pd.DataFrame               # agregasi jenis belanja
    peringatan: pd.DataFrame              # early warning per satker
    tahun_dipakai: list = field(default_factory=list)
    total_pagu: float = 0.0
    total_realisasi: float = 0.0
    total_forecast: float = 0.0
    kebijakan: pd.DataFrame = None


# ---------------------------------------------------------------------------
# Mesin utama
# ---------------------------------------------------------------------------
def hitung_forecast(df: pd.DataFrame, tahun_target: int,
                    scope_satker=None, scope_dept=None,
                    kebijakan: pd.DataFrame = None,
                    ambang_cepat=AMBANG_SERAPAN_CEPAT,
                    ambang_lambat=AMBANG_SERAPAN_LAMBAT,
                    ambang_lonjakan=AMBANG_LONJAKAN_RASIO) -> HasilForecast:
    """Hitung rolling hybrid forecast utk tahun_target pada cakupan terpilih.

    scope_satker : isi utk mengunci ke satu satker (user satker biasa).
    scope_dept   : isi utk mengunci ke satu K/L (mengabaikan scope_dept bila
                   scope_satker diisi).
    """
    kebijakan = KEBIJAKAN_DEFAULT if kebijakan is None else kebijakan
    d = df
    if scope_satker is not None:
        d = d[d["KDSATKER"] == str(scope_satker)]
    elif scope_dept is not None:
        d = d[d["KDDEPT"] == scope_dept]

    df_tahun = d[d["TAHUN"] == tahun_target]
    if df_tahun.empty:
        raise ValueError(f"Tidak ada data tahun {tahun_target} pada cakupan ini.")
    hist = d[(d["TAHUN"] < tahun_target) & (d["TAHUN"] >= tahun_target - 5)]
    tahun_dipakai = sorted(hist["TAHUN"].unique().tolist(), reverse=True)

    bulan_terakhir, k = hitung_bulan_penuh(df_tahun, tahun_target)

    # --- Data tahun berjalan di level kunci detail -------------------------
    cur = agregasi(df_tahun, ["TAHUN"] + KUNCI_DETAIL)
    attr = df_tahun.groupby("KDSATKER", as_index=False).agg(
        KDDEPT=("KDDEPT", "first"), NMDEPT=("NMDEPT", "first"),
        NMSATKER=("NMSATKER", "first"), PROVINSI=("PROVINSI", "first"))
    cur = cur.merge(attr, on="KDSATKER", how="left")

    # --- Profil eksak per kombinasi + fallback berjenjang ------------------
    res_exak = (_profil_tertimbang(agregasi(hist, ["TAHUN"] + KUNCI_DETAIL),
                                   tahun_target, KUNCI_DETAIL)
                if not hist.empty else None)

    hist2 = hist.copy()
    if not hist2.empty:
        ps_hist = hist2.groupby(["TAHUN", "KDSATKER"])["PAGU"].sum().rename("PAGU_SATKER")
        hist2 = hist2.merge(ps_hist, on=["TAHUN", "KDSATKER"], how="left")
        hist2["GRUP_PAGU"] = grup_pagu(hist2["PAGU_SATKER"]).astype(str)
        hist2["_SEMUA_"] = "SEMUA"

    cur["PAGU_SATKER"] = cur["KDSATKER"].map(df_tahun.groupby("KDSATKER")["PAGU"].sum())
    cur["GRUP_PAGU"] = grup_pagu(cur["PAGU_SATKER"]).astype(str)
    cur["_SEMUA_"] = "SEMUA"

    level_fallback = [
        ("historis", KUNCI_DETAIL, res_exak),
        ("sejenis", ["KDDEPT", "JENIS BELANJA", "GRUP_PAGU"], None),
        ("kementerian", ["KDDEPT", "JENIS BELANJA"], None),
        ("jenis", ["JENIS BELANJA"], None),
        ("semua", ["_SEMUA_"], None),
    ]
    for i, (nama, kunci, _) in enumerate(level_fallback):
        if nama == "historis" or hist2.empty:
            continue
        level_fallback[i] = (nama, kunci,
                             _profil_tertimbang(agregasi(hist2, kunci + ["TAHUN"]),
                                                tahun_target, kunci))

    n = len(cur)
    P = pd.DataFrame(np.nan, index=cur.index, columns=BULAN)
    TINGKAT = pd.Series(np.nan, index=cur.index)
    CV = pd.Series(np.nan, index=cur.index)
    W_PROF = pd.Series(np.nan, index=cur.index)
    N_TH = pd.Series(np.nan, index=cur.index)
    METODE = pd.Series("", index=cur.index, dtype=object)
    RP = pd.DataFrame(np.nan, index=cur.index, columns=BULAN)  # rupiah (51)

    for nama, kunci, res in level_fallback:
        if res is None:
            continue
        prof, ting, cvv, w, nth, rp = _sejajarkan(res, cur, kunci)
        mask = P.isna().any(axis=1)
        if not mask.any():
            break
        P.loc[mask, :] = prof.loc[mask].values
        TINGKAT.loc[mask] = ting.loc[mask].values
        CV.loc[mask] = cvv.loc[mask].values
        W_PROF.loc[mask] = w.loc[mask].values
        N_TH.loc[mask] = nth.loc[mask].values
        METODE.loc[mask] = nama
        if nama == "historis":
            RP.loc[mask, :] = rp.loc[mask].values

    # Tanpa histori sama sekali → profil rata & serapan penuh (confidence 0).
    kosong = P.isna().any(axis=1)
    if kosong.any():
        P.loc[kosong, :] = 1.0 / 12.0
        TINGKAT.loc[kosong] = TINGKAT.loc[kosong].fillna(1.0)
        CV.loc[kosong] = np.nan
        METODE.loc[kosong] = "tanpa-histori"

    # --- Target akhir tahun per kombinasi -----------------------------------
    pagu_now = cur["PAGU"].astype(float)
    aktual_k = cur[BULAN[:k]].sum(axis=1) if k else pd.Series(0.0, index=cur.index)
    runrate = aktual_k / k * 12 if k else pd.Series(0.0, index=cur.index)
    is_51 = cur["JENIS BELANJA"] == JB_PEGAWAI

    target_hist_51 = RP.sum(axis=1)                      # rupiah historis (51)
    T = pd.Series(np.nan, index=cur.index)
    T[is_51] = np.maximum.reduce([target_hist_51[is_51].fillna(0.0).values,
                                  runrate[is_51].values,
                                  aktual_k[is_51].values])          # tanpa cap pagu
    t_rate = (TINGKAT * pagu_now).fillna(runrate)
    atas = np.maximum(pagu_now.values, aktual_k.values)             # 52/53: cap pagu
    T[~is_51] = np.clip(np.maximum(t_rate[~is_51].values, aktual_k[~is_51].values),
                        aktual_k[~is_51].values, atas[~is_51])

    # --- Distribusi rolling forecast ke bulan tersisa -----------------------
    sisa_target = (T - aktual_k).clip(lower=0.0)
    tail = P.iloc[:, k:]
    den = tail.sum(axis=1)
    dist = tail.div(den.replace(0, np.nan), axis=0).fillna(1.0 / max(12 - k, 1))
    F = dist.mul(sisa_target, axis=0)

    hasil_bulan = pd.DataFrame(0.0, index=cur.index, columns=BULAN)
    if k:
        hasil_bulan.iloc[:, :k] = cur[BULAN[:k]].astype(float).values
    hasil_bulan.iloc[:, k:] = F.values
    # Jaring pengaman: bulan proyeksi tidak boleh lebih kecil dari realisasi
    # parsial yang sudah tercatat (mis. bulan berjalan).
    hasil_bulan.iloc[:, k:] = np.maximum(hasil_bulan.iloc[:, k:].values,
                                         cur[BULAN[k:]].astype(float).values)

    # --- Policy adjustment Belanja Pegawai (tabel konfigurasi) --------------
    hasil_bulan = _terapkan_kebijakan(cur, hasil_bulan, k, tahun_target, kebijakan)

    # --- Confidence score (dari CV pola historis) ---------------------------
    kelengkapan = (W_PROF.fillna(0.0) / TOTAL_BOBOT).clip(0, 1)
    skor = ((1.0 - CV.fillna(1.0).clip(0, 1)) * 100.0) * (0.6 + 0.4 * kelengkapan)
    # Hanya 1 tahun data → CV antar tahun tak terukur: batasi skor maks 50.
    skor = skor.where(N_TH.fillna(0) >= 2, skor.clip(upper=50.0))
    skor = skor.where(METODE != "tanpa-histori", 0.0).round(1)

    # --- Susun tabel hasil ----------------------------------------------------
    out = cur[["TAHUN", "KDDEPT", "NMDEPT", "KDSATKER", "NMSATKER",
               "PROVINSI", "JENIS BELANJA", "AKUN"]].copy()
    out["PAGU"] = pagu_now
    out[BULAN] = hasil_bulan
    out["REALISASI_AKTUAL"] = aktual_k
    out["TOTAL_FORECAST"] = hasil_bulan.sum(axis=1)
    out["SISA_PAGU"] = pagu_now - aktual_k
    out["METODE"] = METODE
    out["SKOR_CONFIDENCE"] = skor

    # Ekspektasi bulanan historis (utk heatmap & early warning).
    exp = P.multiply(pagu_now * TINGKAT, axis=0)
    mask_rp = is_51 & RP.notna().all(axis=1)
    if mask_rp.any():
        exp.loc[mask_rp, :] = RP.loc[mask_rp].values

    per_satker = out.groupby("KDSATKER", as_index=False).agg(
        NMSATKER=("NMSATKER", "first"), KDDEPT=("KDDEPT", "first"),
        NMDEPT=("NMDEPT", "first"), PROVINSI=("PROVINSI", "first"),
        PAGU=("PAGU", "sum"), REALISASI_AKTUAL=("REALISASI_AKTUAL", "sum"),
        TOTAL_FORECAST=("TOTAL_FORECAST", "sum"),
        **{b: (b, "sum") for b in BULAN})
    per_satker = per_satker.merge(
        exp.groupby("KDSATKER")[BULAN].sum().rename(columns=lambda b: "EXP_" + b),
        on="KDSATKER", how="left")
    per_satker["SKOR_CONFIDENCE"] = (
        out.groupby("KDSATKER")
           .apply(lambda g: np.average(g["SKOR_CONFIDENCE"],
                                       weights=g["PAGU"].replace(0, 1)),
                  include_groups=False).reindex(per_satker["KDSATKER"]).values)

    per_jenis = out.groupby("JENIS BELANJA", as_index=False).agg(
        PAGU=("PAGU", "sum"), REALISASI_AKTUAL=("REALISASI_AKTUAL", "sum"),
        TOTAL_FORECAST=("TOTAL_FORECAST", "sum"),
        **{b: (b, "sum") for b in BULAN})

    peringatan = deteksi_peringatan(per_satker, out, exp, k, bulan_terakhir,
                                    ambang_cepat, ambang_lambat, ambang_lonjakan)

    return HasilForecast(
        tahun=tahun_target, bulan_penuh=k, detail=out, per_satker=per_satker,
        per_jenis=per_jenis, peringatan=peringatan, tahun_dipakai=tahun_dipakai,
        total_pagu=float(pagu_now.sum()),
        total_realisasi=float(aktual_k.sum()),
        total_forecast=float(hasil_bulan.sum().sum()),
        kebijakan=kebijakan,
    )


# ---------------------------------------------------------------------------
# Policy adjustment Belanja Pegawai (dari tabel konfigurasi)
# ---------------------------------------------------------------------------
def _terapkan_kebijakan(cur, hasil_bulan, k, tahun_target, kebijakan):
    """Terapkan penyesuaian kebijakan (mis. kenaikan indeks tukin) ke bulan
    yang masih FORECAST. Baseline = rata-rata realisasi aktual akun tsb pada
    Jan..(mulai−1) yang sudah penuh; kenaikan per bulan =
    baseline × (indeks_baru/indeks_lama − 1). Rapel ditambahkan sekali pada
    bulan rapel utk bulan-bulan efektif yang SUDAH aktual (dibayar indeks lama).
    Angka aktual tidak pernah diubah.
    """
    if kebijakan is None or len(kebijakan) == 0 or k >= 12:
        return hasil_bulan
    hb = hasil_bulan.copy()
    for _, c in kebijakan.iterrows():
        if int(c["tahun"]) != tahun_target:
            continue
        daftar_akun = {str(a).strip().zfill(6)
                       for a in str(c["akun"]).replace(",", ";").split(";") if a.strip()}
        mask = ((cur["KDDEPT"].astype(str) == str(int(c["kddept"])))
                & (cur["AKUN"].isin(daftar_akun))
                & (cur["JENIS BELANJA"] == JB_PEGAWAI))
        if not mask.any():
            continue
        mulai, rapel = int(c["mulai_bulan"]), int(c.get("rapel_bulan") or 0)
        indeks_lama, indeks_baru = float(c["indeks_lama"]), float(c["indeks_baru"])
        if indeks_lama <= 0:
            continue
        n_base = min(mulai - 1, k)
        if n_base <= 0:
            continue  # belum ada aktual sebelum bulan efektif → baseline tak ada
        baseline = cur.loc[mask, BULAN[:n_base]].sum(axis=1) / n_base
        kenaikan = baseline * (indeks_baru / indeks_lama - 1.0)
        idx = cur.index[mask]
        for m in range(k, 12):
            bulan_ke = m + 1
            if bulan_ke < mulai:
                continue
            hb.loc[idx, BULAN[m]] = hb.loc[idx, BULAN[m]] + kenaikan
            if rapel and bulan_ke == rapel:
                tertunda = [b for b in range(mulai, rapel) if b <= k]
                if tertunda:
                    hb.loc[idx, BULAN[m]] = hb.loc[idx, BULAN[m]] + kenaikan * len(tertunda)
    return hb


# ---------------------------------------------------------------------------
# Early warning
# ---------------------------------------------------------------------------
def deteksi_peringatan(per_satker, detail, exp_detail, k, bulan_terakhir,
                       ambang_cepat=AMBANG_SERAPAN_CEPAT,
                       ambang_lambat=AMBANG_SERAPAN_LAMBAT,
                       ambang_lonjakan=AMBANG_LONJAKAN_RASIO) -> pd.DataFrame:
    """Deteksi dini per satker:
    W1 forecast > pagu (non-51) | W2 serapan terlalu cepat | W3 terlalu lambat
    W4 lonjakan tidak wajar bulan terakhir | W5 forecast 51 > pagu (diizinkan,
    hanya info). Skor risiko 0–100 + alasan teks.
    """
    ps = per_satker.copy()
    pagu = ps["PAGU"]
    akt = ps["REALISASI_AKTUAL"]
    fc_total = ps["TOTAL_FORECAST"]

    exp_k = ps[[f"EXP_{b}" for b in BULAN[:k]]].sum(axis=1) if k else pd.Series(0.0, index=ps.index)
    rasio_serapan = akt / exp_k.replace(0, np.nan)

    # W1: porsi forecast non-51 melebihi pagu satker
    lebih = (detail[~(detail["JENIS BELANJA"] == JB_PEGAWAI)]
             .assign(_X=lambda x: x["TOTAL_FORECAST"] - x["PAGU"])
             .loc[lambda x: x["_X"] > 0]
             .groupby("KDSATKER")["_X"].sum())
    w1 = ps["KDSATKER"].map(lebih).fillna(0.0) > 0

    w2 = (rasio_serapan > ambang_cepat) & (exp_k > 0)
    w3 = (rasio_serapan < ambang_lambat) & (exp_k > 0) & (akt >= 0)

    # W4: lonjakan — realisasi bulan penuh terakhir vs ekspektasi bulanan
    w4 = pd.Series(False, index=ps.index)
    if k >= 1:
        bln = BULAN[k - 1]
        real_m = ps[bln]
        exp_m = ps[f"EXP_{bln}"]
        w4 = (real_m > ambang_lonjakan * exp_m) & (exp_m > 0) & (real_m > 0)

    w5 = (fc_total > pagu) & ~w1 & (pagu > 0)  # khusus efek belanja 51

    skor = np.minimum(100, 30 * w1 + 20 * w2 + 20 * w3 + 25 * w4 + 10 * w5).astype(int)
    level = np.select([skor >= 75, skor >= 50, skor >= 25],
                      ["Kritis", "Tinggi", "Waspada"], default="Aman")

    alasan = []
    for i, (s_, r_, a_) in enumerate(zip(w1, rasio_serapan, zip(w2, w3, w4, w5))):
        a = []
        if w1.iloc[i]:
            a.append("Forecast belanja non-pegawai melebihi pagu")
        if w2.iloc[i]:
            a.append(f"Serapan {r_:.0%} dari pola historis (terlalu cepat)")
        if w3.iloc[i]:
            a.append(f"Serapan baru {r_:.0%} dari pola historis (terlalu lambat)")
        if w4.iloc[i]:
            a.append(f"Lonjakan tidak wajar di bulan {BULAN[k-1] if k else '-'}")
        if w5.iloc[i]:
            a.append("Forecast melebihi pagu karena Belanja Pegawai (diizinkan)")
        alasan.append("; ".join(a) if a else "Tidak ada anomali terdeteksi")

    ps["W1_FORECAST_LEBIH_PAGU"] = w1.values
    ps["W2_SERAPAN_CEPAT"] = w2.values
    ps["W3_SERAPAN_LAMBAT"] = w3.values
    ps["W4_LONJAKAN"] = w4.values
    ps["W5_PEGAWAI_LEBIH_PAGU"] = w5.values
    ps["RASIO_SERAPAN_VS_HISTORIS"] = rasio_serapan.round(3)
    ps["SKOR_RISIKO"] = skor
    ps["LEVEL_RISIKO"] = level
    ps["ALASAN"] = alasan
    cols = ["KDSATKER", "NMSATKER", "NMDEPT", "PAGU", "REALISASI_AKTUAL",
            "TOTAL_FORECAST", "RASIO_SERAPAN_VS_HISTORIS",
            "W1_FORECAST_LEBIH_PAGU", "W2_SERAPAN_CEPAT", "W3_SERAPAN_LAMBAT",
            "W4_LONJAKAN", "W5_PEGAWAI_LEBIH_PAGU",
            "SKOR_RISIKO", "LEVEL_RISIKO", "ALASAN"]
    return ps[cols].sort_values("SKOR_RISIKO", ascending=False)


# ---------------------------------------------------------------------------
# Helper tabel (kompatibel dgn tabel per-jenis di view_dashboard_satker.py)
# ---------------------------------------------------------------------------
def tabel_per_jenis(hasil: HasilForecast, label_map: dict = None) -> pd.DataFrame:
    """Tabel jenis belanja × bulan berisi aktual (bulan penuh) + forecast
    (bulan belum penuh) — pengganti gabungan hitung_proyeksi_per_kategori +
    isi_tabel_proyeksi + sesuaikan_proyeksi_tukin_kemenhan.
    """
    t = hasil.per_jenis.set_index("JENIS BELANJA")[BULAN]
    if label_map:
        t.index = [label_map.get(jb, str(jb)) for jb in t.index]
    return t


# ---------------------------------------------------------------------------
# Visualisasi (plotly) — import di dalam fungsi agar mesin bisa dipakai headless
# ---------------------------------------------------------------------------
def grafik_aktual_vs_forecast(hasil: HasilForecast):
    import plotly.graph_objects as go
    total = hasil.detail[BULAN].sum(axis=0)
    k = hasil.bulan_penuh
    aktual = [total.iloc[b - 1] if b <= k else None for b in range(1, 13)]
    proy = []
    for b in range(1, 13):
        if k == 0:
            proy.append(total.iloc[b - 1])
        elif b < k:
            proy.append(None)
        elif b == k:
            proy.append(total.iloc[b - 1])      # titik sambung
        else:
            proy.append(total.iloc[b - 1])
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=BULAN, y=aktual, mode="lines+markers",
                             name="Realisasi per Bulan (Aktual)",
                             line=dict(width=3, shape="spline", smoothing=1.1)))
    fig.add_trace(go.Scatter(x=BULAN, y=proy, mode="lines+markers",
                             name="Forecast (hybrid historis)",
                             line=dict(dash="dash", shape="spline", smoothing=1.1)))
    fig.update_layout(yaxis_title="Rupiah (per bulan)", xaxis_title=None,
                      title="Aktual vs Forecast hingga Akhir Tahun",
                      legend=dict(orientation="h", y=1.12))
    return fig


def grafik_forecast_bulanan(hasil: HasilForecast):
    import plotly.express as px
    k = hasil.bulan_penuh
    total = hasil.detail[BULAN].sum(axis=0)
    df = pd.DataFrame({"Bulan": BULAN[k:],
                       "Forecast": total.values[k:],
                       "Status": "Forecast"})
    if k:
        df_a = pd.DataFrame({"Bulan": BULAN[:k], "Forecast": total.values[:k],
                             "Status": "Aktual"})
        df = pd.concat([df_a, df], ignore_index=True)
    fig = px.bar(df, x="Bulan", y="Forecast", color="Status", text_auto=".2s",
                 color_discrete_map={"Aktual": "#2f6fd0", "Forecast": "#f0a832"},
                 title="Realisasi Aktual & Forecast Bulanan")
    fig.update_layout(yaxis_title="Rupiah", xaxis_title=None)
    return fig


def grafik_waterfall(hasil: HasilForecast):
    import plotly.graph_objects as go
    pagu = hasil.total_pagu
    akt = hasil.total_realisasi
    fc_sisa = hasil.total_forecast - akt
    sisa_akhir = pagu - hasil.total_forecast
    fig = go.Figure(go.Waterfall(
        orientation="v",
        measure=["absolute", "relative", "subtotal", "relative", "total"],
        x=["Pagu", "Realisasi (aktual)", "Sisa Pagu",
           "Forecast sisa tahun", "Sisa setelah Forecast"],
        y=[pagu, -akt, None, -fc_sisa, sisa_akhir],
        textposition="outside",
        decreasing={"marker": {"color": "#e05c5c"}},
        increasing={"marker": {"color": "#4caf7d"}},
        totals={"marker": {"color": "#2f6fd0"}},
    ))
    fig.update_layout(title="Waterfall Pagu – Realisasi – Forecast",
                      yaxis_title="Rupiah")
    return fig


def heatmap_deviasi(hasil: HasilForecast, top_n: int = 20):
    """Heatmap deviasi (aktual/forecast vs ekspektasi historis) antar satker."""
    import plotly.graph_objects as go
    ps = hasil.per_satker.sort_values("PAGU", ascending=False).head(top_n)
    exp_cols = [f"EXP_{b}" for b in BULAN]
    M = ps[BULAN].values.astype(float)
    E = ps[exp_cols].values.astype(float)
    z = np.where(E > 0, (M - E) / E, np.nan)
    labels = [f"{r.KDSATKER} — {str(r.NMSATKER)[:40]}" for r in ps.itertuples()]
    fig = go.Figure(go.Heatmap(
        z=z * 100, x=BULAN, y=labels, colorscale="RdBu_r", zmin=-100, zmax=100,
        hovertemplate="%{y}<br>%{x}: deviasi %{z:.0f}%<extra></extra>",
        colorbar=dict(title="Deviasi %")))
    fig.update_layout(title=f"Heatmap Deviasi vs Pola Historis ({top_n} satker terbesar)",
                      xaxis_title=None, yaxis=dict(autorange="reversed"), height=480)
    return fig


def grafik_confidence(hasil: HasilForecast, top_n: int = 20):
    import plotly.express as px
    ps = (hasil.per_satker.sort_values("PAGU", ascending=False).head(top_n)
          .sort_values("SKOR_CONFIDENCE"))
    ps["LABEL"] = pd.cut(ps["SKOR_CONFIDENCE"], [-1, 40, 60, 80, 101],
                         labels=["Sangat Rendah", "Rendah", "Sedang", "Tinggi"])
    fig = px.bar(ps, x="SKOR_CONFIDENCE",
                 y=[f"{r.KDSATKER} — {str(r.NMSATKER)[:35]}" for r in ps.itertuples()],
                 orientation="h", color="LABEL",
                 color_discrete_map={"Tinggi": "#2e7d32", "Sedang": "#f0a832",
                                     "Rendah": "#ef6c00", "Sangat Rendah": "#c62828"},
                 title="Confidence Score Forecast per Satker (dari CV pola historis)")
    fig.update_layout(xaxis_title="Skor", yaxis_title=None, height=460)
    return fig


def tabel_peringatan_styled(hasil: HasilForecast):
    """Styler tabel early warning siap tampil di Streamlit."""
    p = hasil.peringatan.copy()
    warna = {"Aman": "#e8f5e9", "Waspada": "#fff8e1",
             "Tinggi": "#ffe0b2", "Kritis": "#ffcdd2"}
    styler = (p.style
                .apply(lambda r: [f"background-color:{warna[r['LEVEL_RISIKO']]}"
                                  if c == "LEVEL_RISIKO" else "" for c in r.index], axis=1)
                .format({"PAGU": "Rp {:,.0f}", "REALISASI_AKTUAL": "Rp {:,.0f}",
                         "TOTAL_FORECAST": "Rp {:,.0f}",
                         "RASIO_SERAPAN_VS_HISTORIS": "{:.2f}"}))
    return styler


# ---------------------------------------------------------------------------
# UI Streamlit lengkap (opsional, tinggal panggil dari halaman dashboard)
# ---------------------------------------------------------------------------
def render_forecast_section(hasil: HasilForecast):
    import streamlit as st
    st.subheader("🔮 Forecast Hybrid hingga Akhir Tahun")
    st.caption(
        f"Tahun histori dipakai: {', '.join(map(str, hasil.tahun_dipakai)) or '—'} · "
        f"Bulan penuh terakhir: {BULAN[hasil.bulan_penuh - 1] if hasil.bulan_penuh else '—'} · "
        "Metode: profil tertimbang 50/25/12,5/6,25/6,25 + rolling forecast")

    k1, k2, k3 = st.columns(3)
    k1.metric("Total Forecast Akhir Tahun", f"Rp {hasil.total_forecast:,.0f}",
              f"{hasil.total_forecast / hasil.total_pagu:.1%} dari pagu" if hasil.total_pagu else None)
    k2.metric("Realisasi Aktual", f"Rp {hasil.total_realisasi:,.0f}")
    n_risiko = int((hasil.peringatan["SKOR_RISIKO"] >= 25).sum())
    k3.metric("Satker Berisiko (≥ Waspada)", n_risiko)

    c1, c2 = st.columns(2)
    with c1:
        st.plotly_chart(grafik_aktual_vs_forecast(hasil), use_container_width=True)
    with c2:
        st.plotly_chart(grafik_forecast_bulanan(hasil), use_container_width=True)

    c3, c4 = st.columns(2)
    with c3:
        st.plotly_chart(grafik_waterfall(hasil), use_container_width=True)
    with c4:
        st.plotly_chart(grafik_confidence(hasil), use_container_width=True)

    with st.expander("🌡️ Heatmap deviasi antar satker", expanded=False):
        st.plotly_chart(heatmap_deviasi(hasil), use_container_width=True)

    st.markdown("### ⚠️ Early Warning")
    st.dataframe(tabel_peringatan_styled(hasil), use_container_width=True)
    st.caption(
        "W1: forecast non-pegawai > pagu · W2/W3: serapan terlalu cepat/lambat "
        "vs pola historis · W4: lonjakan tidak wajar · W5: belanja pegawai "
        "melebihi pagu (diizinkan sesuai kebijakan).")


# ---------------------------------------------------------------------------
# Uji cepat: python forecast.py
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    df = muat_data("sample pagu realisasi.csv")
    hasil = hitung_forecast(df, tahun_target=2026)
    print("bulan_penuh:", hasil.bulan_penuh)
    print(f"pagu={hasil.total_pagu:,.0f} aktual={hasil.total_realisasi:,.0f} "
          f"forecast={hasil.total_forecast:,.0f}")
    print(hasil.per_jenis[["JENIS BELANJA", "PAGU", "REALISASI_AKTUAL", "TOTAL_FORECAST"]])
    print(hasil.peringatan[["KDSATKER", "SKOR_RISIKO", "LEVEL_RISIKO", "ALASAN"]].head())