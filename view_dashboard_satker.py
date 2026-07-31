import streamlit as st
import pandas as pd
from datetime import datetime

# Asumsi bulan berjalan didapatkan dari waktu saat ini atau parameter aplikasi
bulan_saat_ini = datetime.now().month

def highlight_proyeksi(row, bulan_berjalan):
    """Fungsi pewarnaan: baris pada bulan berjalan dan akan datang diwarnai kuning"""
    # Asumsi row.name adalah index yang melambangkan bulan (1-12)
    if row.name >= bulan_berjalan:
        return ['background-color: yellow'] * len(row)
    return [''] * len(row)

def render_tabel_realisasi(df_tabel_realisasi):
    st.subheader("Tabel Realisasi Bulanan per Jenis Belanja")
    
    # 1. Terapkan styler pada dataframe untuk memberikan warna kuning
    df_styled = df_tabel_realisasi.style.apply(highlight_proyeksi, bulan_berjalan=bulan_saat_ini, axis=1)
    
    st.dataframe(df_styled)
    
    # 2. Keterangan cara perhitungan proyeksi tidak dicantumkan, cukup keterangan sel kuning
    st.caption("Sel berwarna kuning = mengandung angka proyeksi (bulan yang belum berakhir)")

def render_grafik_tren(fig_tren_proyeksi):
    st.subheader("Tren & Proyeksi Realisasi hingga Akhir Tahun")
    
    # Menampilkan grafik
    st.plotly_chart(fig_tren_proyeksi, use_container_width=True)
    
    # 3. Grafik Tren & Proyeksi Realisasi tidak diberi keterangan cara perhitungan
    # (Kode st.write / st.caption mengenai perhitungan sebelumnya telah dihapus)
