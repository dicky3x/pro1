"""
Halaman 4: Dashboard Dana Transfer ke Daerah (TKD)
-----------------------------------------------------
Memakai sumber data yang SAMA dengan Halaman 1 (data/pagu_realisasi.csv.gz), difilter
hanya ke kategori Transfer ke Daerah (kode jenis belanja 61-66: DBH, DAU, DAK Fisik,
DAK Nonfisik, Insentif Fiskal, Dana Desa). Karena TKD disalurkan oleh KPPN untuk
kabupaten/kota (bukan per satker K/L), halaman ini TIDAK dibatasi ke satker milik
user yang login seperti Halaman 1 -- semua user yang login bisa lihat gambaran TKD
se-provinsi, dan filter utamanya adalah Kabupaten/Kota (bukan Satker).
"""

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from common import (
    BULAN_KOLOM, BULAN_LABEL, get_data, tanggal_update_data, require_login, kpi_card,
    hitung_proyeksi_agregat, hitung_proyeksi_per_kategori, isi_tabel_proyeksi,
    hitung_bulan_penuh_terakhir, KODE_JENIS_TKD,
)

st.set_page_config(
    page_title="DATUK - Dashboard Dana Transfer ke Daerah",
    page_icon="🏘️",
    layout="wide",
)

df_semua = get_data()
auth = require_login(df_semua, judul_halaman="DATUK")

if "KABKOTA" not in df_semua.columns:
    st.error(
        "Kolom KABKOTA tidak ditemukan di data. Halaman ini butuh kolom kabupaten/kota "
        "untuk membagi tampilan Dana Transfer ke Daerah."
    )
    st.stop()

df = df_semua[df_semua["JENIS BELANJA"].isin(KODE_JENIS_TKD)]

if df.empty:
    st.warning("Tidak ada data Transfer ke Daerah (kode jenis belanja 61-66) di dataset ini.")
    st.stop()


# --------------------------------------------------------------------------
# Sidebar - filter
# --------------------------------------------------------------------------

st.sidebar.header("Filter")

tahun_list = sorted(df["TAHUN"].unique(), reverse=True)
tahun = st.sidebar.selectbox("Tahun", tahun_list)
df_tahun = df[df["TAHUN"] == tahun]

SEMUA_KABKOTA = "— Semua Kabupaten/Kota —"
kabkota_list = sorted(df_tahun["KABKOTA"].dropna().unique().tolist())
kabkota_pilih = st.sidebar.selectbox("Kabupaten/Kota", [SEMUA_KABKOTA] + kabkota_list)

if kabkota_pilih == SEMUA_KABKOTA:
    kabkota = None
    df_wilayah = df_tahun
else:
    kabkota = kabkota_pilih
    df_wilayah = df_tahun[df_tahun["KABKOTA"] == kabkota]


# --------------------------------------------------------------------------
# Agregasi
# --------------------------------------------------------------------------

pagu_total = df_wilayah["PAGU"].sum()
realisasi_total = df_wilayah["REALISASI"].sum()
sisa_pagu = df_wilayah["SISA PAGU"].sum()
persen_serapan = (realisasi_total / pagu_total * 100) if pagu_total else 0

monthly = df_wilayah[BULAN_KOLOM].sum()
kumulatif = monthly.cumsum()

bulan_terakhir, bulan_penuh_terakhir = hitung_bulan_penuh_terakhir(df_wilayah, tahun)

jenis_tkd = (
    df_wilayah.groupby("LABEL_JENIS_BELANJA")["REALISASI"]
    .sum()
    .sort_values(ascending=False)
    .reset_index()
)


# --------------------------------------------------------------------------
# Proyeksi (rumus & bobot sama seperti Halaman 1, lihat common.py)
# --------------------------------------------------------------------------

FILTER_ENTITAS = {"KABKOTA": kabkota}

proyeksi_agregat_bulanan, tahun_dipakai = hitung_proyeksi_agregat(df, tahun, pagu_total, FILTER_ENTITAS)

if proyeksi_agregat_bulanan is None:
    rerata_bulanan = (kumulatif.iloc[bulan_terakhir - 1] / bulan_terakhir) if bulan_terakhir else 0
    proyeksi_akhir_tahun = rerata_bulanan * 12
    metode_proyeksi = "fallback"
else:
    target_tahun_penuh = proyeksi_agregat_bulanan.sum()
    proyeksi_akhir_tahun = max(realisasi_total, target_tahun_penuh)
    metode_proyeksi = "historis"

persen_proyeksi = (proyeksi_akhir_tahun / pagu_total * 100) if pagu_total else 0


# --------------------------------------------------------------------------
# Header & KPI
# --------------------------------------------------------------------------

st.title("🏘️ Dashboard Dana Transfer ke Daerah (TKD)")
st.caption(f"🕒 Data terakhir diperbarui: {tanggal_update_data()}")
st.caption(f"{kabkota or 'Semua Kabupaten/Kota'} — Tahun {tahun}")

r1c1, r1c2 = st.columns(2)
r2c1, r2c2 = st.columns(2)

with r1c1:
    kpi_card("Pagu TKD", f"Rp {pagu_total:,.0f}")
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

# --------------------------------------------------------------------------
# Grafik batang
# --------------------------------------------------------------------------

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


# --------------------------------------------------------------------------
# Pie charts
# --------------------------------------------------------------------------

st.subheader("Komposisi")
p1, p2 = st.columns(2)

with p1:
    st.caption("Realisasi vs Sisa Pagu")
    fig_pie1 = px.pie(
        names=["Realisasi", "Sisa Pagu"],
        values=[realisasi_total, max(sisa_pagu, 0)],
        hole=0.4,
    )
    st.plotly_chart(fig_pie1, use_container_width=True)

with p2:
    st.caption("Realisasi per Jenis TKD")
    fig_pie2 = px.pie(jenis_tkd, names="LABEL_JENIS_BELANJA", values="REALISASI", hole=0.4)
    st.plotly_chart(fig_pie2, use_container_width=True)


# --------------------------------------------------------------------------
# Perbandingan antar kabupaten/kota (hanya relevan kalau "Semua" yang dipilih)
# --------------------------------------------------------------------------

if kabkota is None:
    st.subheader("Perbandingan antar Kabupaten/Kota")
    per_kabkota = (
        df_tahun.groupby("KABKOTA")
        .agg(Pagu=("PAGU", "sum"), Realisasi=("REALISASI", "sum"))
        .reset_index()
        .sort_values("Pagu", ascending=False)
    )
    per_kabkota["Persen Realisasi"] = (
        per_kabkota["Realisasi"] / per_kabkota["Pagu"].replace(0, np.nan) * 100
    ).fillna(0)

    fig_kabkota = go.Figure(data=[
        go.Bar(name="Pagu", x=per_kabkota["KABKOTA"], y=per_kabkota["Pagu"]),
        go.Bar(name="Realisasi", x=per_kabkota["KABKOTA"], y=per_kabkota["Realisasi"]),
    ])
    fig_kabkota.update_layout(barmode="group", yaxis_title="Rupiah", xaxis_title=None)
    st.plotly_chart(fig_kabkota, use_container_width=True)

    st.dataframe(
        per_kabkota.style.format({
            "Pagu": "Rp {:,.0f}", "Realisasi": "Rp {:,.0f}", "Persen Realisasi": "{:.1f}%",
        }),
        use_container_width=True,
        hide_index=True,
    )
    st.divider()


# --------------------------------------------------------------------------
# Trendline proyeksi
# --------------------------------------------------------------------------

st.subheader("Tren & Proyeksi Realisasi hingga Akhir Tahun")

bulan_angka = list(range(1, 13))


def _nilai_proyeksi_bulan(b):
    if proyeksi_agregat_bulanan is not None:
        return proyeksi_agregat_bulanan[b - 1]
    return rerata_bulanan


aktual = [monthly.values[b - 1] if b <= bulan_penuh_terakhir else None for b in bulan_angka]
proyeksi = []
for b in bulan_angka:
    if bulan_penuh_terakhir == 0:
        proyeksi.append(_nilai_proyeksi_bulan(b))
    elif b < bulan_penuh_terakhir:
        proyeksi.append(None)
    elif b == bulan_penuh_terakhir:
        proyeksi.append(monthly.values[b - 1])
    else:
        proyeksi.append(_nilai_proyeksi_bulan(b))

fig_trend = go.Figure()
fig_trend.add_trace(go.Scatter(
    x=BULAN_KOLOM, y=aktual, mode="lines+markers",
    name="Realisasi per Bulan (Aktual)",
    line=dict(width=3, shape="spline", smoothing=1.1),
))
fig_trend.add_trace(go.Scatter(
    x=BULAN_KOLOM, y=proyeksi, mode="lines+markers",
    name="Proyeksi (rata-rata bulanan)",
    line=dict(dash="dash", shape="spline", smoothing=1.1),
))
fig_trend.update_layout(yaxis_title="Rupiah (per bulan)", xaxis_title=None)
st.plotly_chart(fig_trend, use_container_width=True)

_label_batas = BULAN_LABEL.get(bulan_penuh_terakhir, "-") if bulan_penuh_terakhir else None
_batas_teks = f"Bulan setelah {_label_batas}" if _label_batas else "Seluruh bulan tahun ini"

if metode_proyeksi == "historis":
    daftar_tahun_ket = ", ".join(str(t) for t in tahun_dipakai)
    st.caption(
        "Grafik ini menampilkan realisasi TKD tiap bulan (bukan kumulatif). "
        f"{_batas_teks} adalah proyeksi yang dihitung dari rerata tertimbang tingkat realisasi "
        f"tahun-tahun sebelumnya dikalikan pagu tahun {tahun}. Tahun historis yang dipakai: "
        f"{daftar_tahun_ket}."
    )
else:
    st.caption(
        "Grafik ini menampilkan realisasi TKD tiap bulan (bukan kumulatif). "
        f"{_batas_teks} adalah proyeksi. Belum ada data historis (tahun sebelum {tahun}) untuk "
        "wilayah ini, sehingga proyeksi memakai metode cadangan: rata-rata realisasi tahun berjalan."
    )


# --------------------------------------------------------------------------
# Tabel realisasi per bulan per jenis TKD
# --------------------------------------------------------------------------

st.markdown("**Realisasi Bulanan per Jenis Transfer ke Daerah**")

_tgl_update_lengkap = tanggal_update_data()
_tgl_saja = _tgl_update_lengkap.split(",")[0].strip() if "," in _tgl_update_lengkap else _tgl_update_lengkap

BARIS_TOTAL_REALISASI_RP = f"Total Realisasi (Rp) sd. tanggal {_tgl_saja}"
BARIS_TOTAL_REALISASI_PCT = f"Total Realisasi (%) sd. tanggal {_tgl_saja}"
BARIS_SISA_PAGU = "Sisa Pagu"
BARIS_TOTAL_PROYEKSI_RP = "Total Realisasi + Proyeksi Akhir Tahun (Rp)"
BARIS_TOTAL_PROYEKSI_PCT = "Total Realisasi + Proyeksi Akhir Tahun (%)"

urutan_kode = (
    df_wilayah[["JENIS BELANJA", "LABEL_JENIS_BELANJA"]]
    .drop_duplicates()
    .sort_values("JENIS BELANJA")["LABEL_JENIS_BELANJA"]
    .tolist()
)

pagu_per_jenis = df_wilayah.groupby("LABEL_JENIS_BELANJA")["PAGU"].sum().reindex(urutan_kode)

realisasi_aktual_jenis = (
    df_wilayah.groupby("LABEL_JENIS_BELANJA")[BULAN_KOLOM].sum()
    .astype(float)
    .reindex(urutan_kode)
)

proyeksi_per_jenis, _ = hitung_proyeksi_per_kategori(
    df, tahun, pagu_per_jenis.reindex(urutan_kode), FILTER_ENTITAS, "LABEL_JENIS_BELANJA"
)

# Beda dengan Halaman 1: TIDAK ada kategori yang dikecualikan dari cap 100% di sini (semua
# jenis TKD tunduk pada pagu masing-masing, tidak ada yang setara "Belanja Pegawai").
tabel_tampil = isi_tabel_proyeksi(
    realisasi_aktual_jenis, proyeksi_per_jenis, pagu_per_jenis, bulan_penuh_terakhir,
    kategori_dikecualikan_cap=None,
)

tabel_t = tabel_tampil.reindex(urutan_kode).T
tabel_t["TOTAL"] = tabel_t.sum(axis=1)

pagu_row = pagu_per_jenis.reindex(urutan_kode).copy()
pagu_row["TOTAL"] = pagu_total
pagu_row.name = "PAGU"

pagu_aman = pagu_per_jenis.reindex(urutan_kode).replace(0, np.nan)

total_real_rp = realisasi_aktual_jenis.sum(axis=1)
total_real_rp["TOTAL"] = total_real_rp.sum()
total_real_rp.name = BARIS_TOTAL_REALISASI_RP

total_real_pct = (realisasi_aktual_jenis.sum(axis=1) / pagu_aman * 100).fillna(0)
total_real_pct["TOTAL"] = (total_real_rp["TOTAL"] / pagu_total * 100) if pagu_total else 0
total_real_pct.name = BARIS_TOTAL_REALISASI_PCT

sisa_pagu_row = (pagu_per_jenis.reindex(urutan_kode) - realisasi_aktual_jenis.sum(axis=1))
sisa_pagu_row["TOTAL"] = pagu_total - total_real_rp["TOTAL"]
sisa_pagu_row.name = BARIS_SISA_PAGU

total_proyeksi_rp = tabel_tampil.reindex(urutan_kode).sum(axis=1)
total_proyeksi_rp["TOTAL"] = total_proyeksi_rp.sum()
total_proyeksi_rp.name = BARIS_TOTAL_PROYEKSI_RP

total_proyeksi_pct = (tabel_tampil.reindex(urutan_kode).sum(axis=1) / pagu_aman * 100).fillna(0)
total_proyeksi_pct["TOTAL"] = (total_proyeksi_rp["TOTAL"] / pagu_total * 100) if pagu_total else 0
total_proyeksi_pct.name = BARIS_TOTAL_PROYEKSI_PCT

tabel_final = pd.concat([
    pagu_row.to_frame().T,
    tabel_t,
    total_real_rp.to_frame().T,
    total_real_pct.to_frame().T,
    sisa_pagu_row.to_frame().T,
    total_proyeksi_rp.to_frame().T,
    total_proyeksi_pct.to_frame().T,
])
tabel_final = tabel_final.reindex(columns=urutan_kode + ["TOTAL"])

BARIS_RUPIAH = ["PAGU"] + BULAN_KOLOM + [BARIS_TOTAL_REALISASI_RP, BARIS_SISA_PAGU, BARIS_TOTAL_PROYEKSI_RP]
BARIS_PERSEN = [BARIS_TOTAL_REALISASI_PCT, BARIS_TOTAL_PROYEKSI_PCT]

baris_bulan_proyeksi = [b for i, b in enumerate(BULAN_KOLOM) if i >= bulan_penuh_terakhir]
mask_final = pd.DataFrame(False, index=tabel_final.index, columns=tabel_final.columns)
mask_final.loc[baris_bulan_proyeksi, :] = True
if bulan_penuh_terakhir < 12:
    mask_final.loc[BARIS_TOTAL_PROYEKSI_RP, :] = True
    mask_final.loc[BARIS_TOTAL_PROYEKSI_PCT, :] = True


def _style_tabel(_):
    styles = pd.DataFrame(
        np.where(mask_final, "background-color: #fff3cd; color: #7a5b00;", ""),
        index=mask_final.index, columns=mask_final.columns,
    )
    styles.loc["PAGU", :] = styles.loc["PAGU", :] + "font-weight: bold;"
    return styles


styled_tabel = (
    tabel_final.style
    .apply(_style_tabel, axis=None)
    .format("Rp {:,.0f}", subset=pd.IndexSlice[BARIS_RUPIAH, :])
    .format("{:.1f}%", subset=pd.IndexSlice[BARIS_PERSEN, :])
)
st.dataframe(styled_tabel, use_container_width=True)
st.caption(
    "🟨 Sel berwarna kuning = mengandung angka proyeksi (bulan yang belum berakhir). Baris "
    "\"Total Realisasi\" hanya menjumlahkan uang yang sudah benar-benar tersalurkan (bulan "
    "penuh saja), sedangkan baris \"Total Realisasi + Proyeksi Akhir Tahun\" menjumlahkan "
    "realisasi ditambah estimasi bulan-bulan yang belum berakhir/belum terjadi."
)
