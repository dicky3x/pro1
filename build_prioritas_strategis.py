"""
build_prioritas_strategis.py
-------------------------------
Mengubah file xlsx "Prioritas Presiden" dan "Program Strategis" (struktur sumbernya
sama persis, cuma beda nama kolom kategorinya) menjadi CSV terkompresi yang dipakai
Halaman 2 & 3.

Beda dengan pagu_realisasi.csv.gz: file-file ini HANYA berisi baris yang sudah
ditandai prioritas/strategis (bukan seluruh data satker), dan HANYA untuk tahun 2026
(tidak ada histori tahun sebelumnya) -- proyeksi akhir tahun otomatis fallback ke
metode rata-rata bulan berjalan (lihat common.py::hitung_proyeksi_agregat).

Jalankan:
    python build_prioritas_strategis.py
"""

import pandas as pd

BULAN_SRC = ["jan", "feb", "mar", "apr", "mei", "jun", "jul", "ags", "sep", "okt", "nov", "des"]
BULAN_OUT = ["JAN", "FEB", "MAR", "APR", "MEI", "JUN", "JUL", "AGS", "SEP", "OKT", "NOV", "DES"]


def konversi(src_path: str, sheet_name: str, kolom_kode_kategori: str, kolom_nama_kategori: str,
             out_path: str):
    df = pd.read_excel(src_path, sheet_name=sheet_name)
    df.columns = [c.strip() for c in df.columns]

    out = pd.DataFrame()
    out["TAHUN"] = df["TAHUN"].astype(int)
    out["KDDEPT"] = df["kementerian_kode"].astype(int)
    out["NMDEPT"] = df["kementerian_uraian"].astype(str).str.strip()
    out["KDSATKER"] = df["satker_kode"].astype(int)
    out["NMSATKER"] = df["satker_uraian"].astype(str).str.strip()
    out["PROVINSI"] = df["provinsi_uraian"].astype(str).str.strip()
    out["KABKOTA"] = df["kabkota_uraian"].astype(str).str.strip()
    out["FUNGSI"] = df["fungsi_uraian"].astype(str).str.strip()
    out["SUBFUNGSI"] = df["subfungsi_uraian"].astype(str).str.strip()
    out["PROGRAM"] = df["program_uraian"].astype(str).str.strip()
    out["KEGIATAN_KODE"] = df["kegiatan_kode"].astype(str).str.strip()
    out["KEGIATAN"] = df["kegiatan_uraian"].astype(str).str.strip()
    out["OUTPUT_KODE"] = df["outputkro_kode"].astype(str).str.strip()
    out["OUTPUT"] = df["outputkro_uraian"].astype(str).str.strip()
    out["SUBOUTPUT_KODE"] = df["suboutputro_kode"].astype(str).str.strip()
    out["SUBOUTPUT"] = df["suboutputro_uraian"].astype(str).str.strip()
    out["AKUN"] = df["akun_uraian"].astype(str).str.strip()

    out["KATEGORI_KODE"] = df[kolom_kode_kategori].astype(str).str.strip()
    out["KATEGORI"] = df[kolom_nama_kategori].astype(str).str.strip()

    out["PAGU"] = pd.to_numeric(df["pagu_dipa"], errors="coerce").fillna(0)
    for src, dst in zip(BULAN_SRC, BULAN_OUT):
        out[dst] = pd.to_numeric(df[src], errors="coerce").fillna(0) if src in df.columns else 0

    out["BLOKIR"] = pd.to_numeric(df.get("blokir", 0), errors="coerce").fillna(0)

    out["REALISASI"] = out[BULAN_OUT].sum(axis=1)
    out["SISA PAGU"] = out["PAGU"] - out["REALISASI"]

    out.to_csv(out_path, index=False, compression="gzip")
    print(f"Selesai: {len(out):,} baris -> {out_path}")
    print(f"  Kategori: {sorted(out['KATEGORI'].unique())}")
    print(f"  Tahun: {sorted(out['TAHUN'].unique())}")


if __name__ == "__main__":
    konversi(
        "/mnt/user-data/uploads/prioritas_presiden_2026.xlsx",
        "prioritas presiden 2026",
        "jenisprioritaspresiden_kode",
        "nama prioritas presiden",
        "data/prioritas_presiden.csv.gz",
    )
    konversi(
        "/mnt/user-data/uploads/program_strategis2026.xlsx",
        "program strategis2026",
        "jenisprogramstrategis_kode",
        "nama program strategis",
        "data/program_strategis.csv.gz",
    )
