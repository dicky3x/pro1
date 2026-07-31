import pandas as pd
import numpy as np
from datetime import datetime

def hitung_proyeksi_berbobot(df_historis, realisasi_tahun_ini, bulan_berjalan):
    """
    Menghitung proyeksi dengan bobot:
    (Y-1 x 50%) + (Y-2 x 25%) + (Y-3 x 12.5%) + (Y-4 x 6.25%) + (Y-5 x 6.25%)
    
    df_historis: DataFrame dengan index bulan (1-12) dan kolom ['Y-1', 'Y-2', 'Y-3', 'Y-4', 'Y-5']
    realisasi_tahun_ini: Dictionary/Series berisi realisasi per bulan
    bulan_berjalan: Integer (1-12)
    """
    bobot = {'Y-1': 0.5, 'Y-2': 0.25, 'Y-3': 0.125, 'Y-4': 0.0625, 'Y-5': 0.0625}
    proyeksi_akhir = {}
    
    for bulan in range(1, 13):
        if bulan < bulan_berjalan:
            # Untuk bulan yang sudah berakhir, nilainya sesuai realisasi
            proyeksi_akhir[bulan] = realisasi_tahun_ini.get(bulan, 0)
        else:
            # Hitung pembobotan proporsional (berdasarkan rata-rata 5 tahun)
            nilai_proyeksi = 0
            for thn, b in bobot.items():
                if thn in df_historis.columns:
                    # Nilai ini bisa berupa persentase atau nominal
                    nilai_proyeksi += df_historis.loc[bulan, thn] * b
            
            if bulan == bulan_berjalan:
                # Untuk bulan berjalan, pastikan proyeksinya tidak lebih kecil dari realisasi hingga hari ini
                realisasi_hari_ini = realisasi_tahun_ini.get(bulan, 0)
                proyeksi_akhir[bulan] = max(nilai_proyeksi, realisasi_hari_ini)
            else:
                # Untuk bulan yang akan datang, berikan pembobotan proporsional
                proyeksi_akhir[bulan] = nilai_proyeksi
                
    return proyeksi_akhir
