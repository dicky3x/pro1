import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

# --------------------------------------------------------------------------
# KONFIGURASI KEBIJAKAN (POLICY ADJUSTMENT) - CONTOH FORMAT
# Aplikasi frontend dapat memodifikasi dataframe ini tanpa mengubah kode inti.
# --------------------------------------------------------------------------
DEFAULT_POLICY_CONFIG = pd.DataFrame([
    {
        "KDDEPT": 12, # Kementerian Pertahanan
        "AKUN_PREFIX": "512411", # Tunjangan Kinerja
        "BULAN_MULAI": 7, # Juli
        "PERSEN_KENAIKAN": 0.28, # 28%
        "BULAN_RAPEL": 9 # Dirapel di September
    }
])

# Bobot Historis (Tahun ke-1 mundur sampai ke-5)
BOBOT_TAHUN_HISTORIS = {1: 0.50, 2: 0.25, 3: 0.125, 4: 0.0625, 5: 0.0625}

# --------------------------------------------------------------------------
# ENGINE PROYEKSI & EARLY WARNING
# --------------------------------------------------------------------------

def _kategorikan_pagu(pagu):
    """Membagi pagu ke dalam kelompok (Binning) untuk keperluan fallback profile"""
    if pagu < 100_000_000: return "KECIL"
    if pagu < 1_000_000_000: return "MENENGAH"
    if pagu < 10_000_000_000: return "BESAR"
    return "SANGAT BESAR"

def buat_profil_historis(df_all: pd.DataFrame, tahun_berjalan: int) -> tuple:
    """Membentuk rasio pencairan bulanan (spending profile) menggunakan weighted average"""
    df_hist = df_all[(df_all['TAHUN'] >= tahun_berjalan - 5) & (df_all['TAHUN'] < tahun_berjalan)].copy()
    
    if df_hist.empty:
        return None, None, None

    # Tentukan Bobot berdasarkan Tahun
    df_hist['TAHUN_MUNDUR'] = tahun_berjalan - df_hist['TAHUN']
    df_hist['BOBOT'] = df_hist['TAHUN_MUNDUR'].map(BOBOT_TAHUN_HISTORIS)
    
    # Hitung persentase realisasi per bulan terhadap total realisasi setahun
    df_hist['TOTAL_REALISASI_THN'] = df_hist[BULAN_KOLOM].sum(axis=1)
    df_hist = df_hist[df_hist['TOTAL_REALISASI_THN'] > 0] # Abaikan yang 0
    
    for bln in BULAN_KOLOM:
        df_hist[f'PCT_{bln}'] = df_hist[bln] / df_hist['TOTAL_REALISASI_THN']
        df_hist[f'W_PCT_{bln}'] = df_hist[f'PCT_{bln}'] * df_hist['BOBOT']

    # 1. Profil Level Satker - Jenis Belanja - Akun
    profil_satker = df_hist.groupby(['KDSATKER', 'JENIS BELANJA', 'AKUN']).apply(
        lambda x: pd.Series({
            f'PROF_{bln}': x[f'W_PCT_{bln}'].sum() / x['BOBOT'].sum() for bln in BULAN_KOLOM
        })
    ).reset_index()

    # Hitung Confidence Score (Coefficient of Variation) di level Satker
    cv_satker = df_hist.groupby(['KDSATKER', 'JENIS BELANJA', 'AKUN'])[BULAN_KOLOM].std().sum(axis=1) / \
                (df_hist.groupby(['KDSATKER', 'JENIS BELANJA', 'AKUN'])[BULAN_KOLOM].mean().sum(axis=1) + 1e-9)
    profil_satker['CONFIDENCE_SCORE'] = 1 - cv_satker.clip(upper=1) # 1 = Sangat Konsisten, 0 = Volatil

    # 2. Profil Level Kementerian (Fallback 1)
    df_hist['KELOMPOK_PAGU'] = df_hist['PAGU'].apply(_kategorikan_pagu)
    profil_dept = df_hist.groupby(['KDDEPT', 'JENIS BELANJA', 'KELOMPOK_PAGU']).apply(
        lambda x: pd.Series({
            f'PROF_{bln}': x[f'W_PCT_{bln}'].sum() / x['BOBOT'].sum() for bln in BULAN_KOLOM
        })
    ).reset_index()

    # 3. Profil Nasional (Fallback 2)
    profil_nasional = df_hist.groupby(['JENIS BELANJA']).apply(
        lambda x: pd.Series({
            f'PROF_{bln}': x[f'W_PCT_{bln}'].sum() / x['BOBOT'].sum() for bln in BULAN_KOLOM
        })
    ).reset_index()

    return profil_satker, profil_dept, profil_nasional

def hitung_hybrid_forecast(df_current: pd.DataFrame, profil_satker: pd.DataFrame, 
                           profil_dept: pd.DataFrame, profil_nasional: pd.DataFrame, 
                           bulan_penuh_terakhir: int, policy_config: pd.DataFrame = DEFAULT_POLICY_CONFIG) -> pd.DataFrame:
    """Menghitung Rolling Forecast ke akhir tahun dengan hybrid fallback & policy adjustment"""
    
    df_fc = df_current.copy()
    df_fc['KELOMPOK_PAGU'] = df_fc['PAGU'].apply(_kategorikan_pagu)
    
    # --- STEP 1: Penggabungan Profil Historis (Fallback Mechanism) ---
    kolom_prof = [f'PROF_{b}' for b in BULAN_KOLOM]
    
    if profil_satker is not None:
        df_fc = df_fc.merge(profil_satker, on=['KDSATKER', 'JENIS BELANJA', 'AKUN'], how='left')
        df_fc = df_fc.merge(profil_dept, on=['KDDEPT', 'JENIS BELANJA', 'KELOMPOK_PAGU'], how='left', suffixes=('', '_DEPT'))
        df_fc = df_fc.merge(profil_nasional, on=['JENIS BELANJA'], how='left', suffixes=('', '_NAS'))
        
        # Terapkan fallback berjenjang
        for bln in BULAN_KOLOM:
            prof_col = f'PROF_{bln}'
            df_fc[prof_col] = df_fc[prof_col].fillna(df_fc[f'{prof_col}_DEPT']).fillna(df_fc[f'{prof_col}_NAS'])
            
        df_fc.drop(columns=[c for c in df_fc.columns if c.endswith('_DEPT') or c.endswith('_NAS')], inplace=True)
    else:
        # Jika tidak ada histori sama sekali (rata-rata dibagi rata 1/12)
        for bln in BULAN_KOLOM:
            df_fc[f'PROF_{bln}'] = 1.0 / 12.0
            
    df_fc['CONFIDENCE_SCORE'] = df_fc.get('CONFIDENCE_SCORE', 0.5).fillna(0.3) # Penalti skor jika pakai fallback

    # --- STEP 2: Rolling Forecast ---
    bulan_sisa = BULAN_KOLOM[bulan_penuh_terakhir:]
    bulan_aktual = BULAN_KOLOM[:bulan_penuh_terakhir]
    
    df_fc['REALISASI_YTD'] = df_fc[bulan_aktual].sum(axis=1) if bulan_aktual else 0
    df_fc['SISA_PAGU'] = np.maximum(df_fc['PAGU'] - df_fc['REALISASI_YTD'], 0)
    
    # Hitung bobot profil yang tersisa di sisa bulan
    df_fc['SISA_BOBOT_PROFIL'] = df_fc[[f'PROF_{b}' for b in bulan_sisa]].sum(axis=1)
    
    for i, bln in enumerate(bulan_sisa):
        idx_bulan = bulan_penuh_terakhir + i + 1
        
        # Normalisasi profil ke sisa bulan
        faktor_distribusi = np.where(
            df_fc['SISA_BOBOT_PROFIL'] > 0,
            df_fc[f'PROF_{bln}'] / df_fc['SISA_BOBOT_PROFIL'],
            1.0 / len(bulan_sisa) # Fallback flat jika bobot sisa 0
        )
        
        # Default proyeksi: Sisa Pagu * Faktor Distribusi
        proyeksi = df_fc['SISA_PAGU'] * faktor_distribusi
        
        # --- PERLAKUAN KHUSUS BELANJA PEGAWAI (51) ---
        # Belanja 51 tidak dibatasi pagu. Target akhir tahun disesuaikan dengan run-rate YTD.
        is_pegawai = df_fc['JENIS BELANJA'] == 51
        if is_pegawai.any() and bulan_penuh_terakhir > 0:
            run_rate_bulanan = df_fc.loc[is_pegawai, 'REALISASI_YTD'] / bulan_penuh_terakhir
            target_tahunan = np.maximum(df_fc.loc[is_pegawai, 'PAGU'], run_rate_bulanan * 12)
            sisa_target_51 = np.maximum(target_tahunan - df_fc.loc[is_pegawai, 'REALISASI_YTD'], 0)
            proyeksi[is_pegawai] = sisa_target_51 * faktor_distribusi[is_pegawai]

        # --- POLICY ADJUSTMENTS ---
        for _, policy in policy_config.iterrows():
            mask_policy = (df_fc['KDDEPT'] == policy['KDDEPT']) & (df_fc['AKUN'].astype(str).str.startswith(policy['AKUN_PREFIX']))
            
            if mask_policy.any():
                # Baseline diambil dari rata-rata YTD sebelum policy berlaku
                baseline = (df_fc.loc[mask_policy, BULAN_KOLOM[:policy['BULAN_MULAI']-1]].sum(axis=1) / (policy['BULAN_MULAI']-1)).fillna(0)
                tambahan = baseline * policy['PERSEN_KENAIKAN']
                
                # Tambahkan kenaikan rutin
                if idx_bulan >= policy['BULAN_MULAI']:
                    proyeksi[mask_policy] += tambahan
                
                # Hitung dan terapkan rapel
                if idx_bulan == policy['BULAN_RAPEL']:
                    bulan_terlewat = min(bulan_penuh_terakhir, policy['BULAN_RAPEL'] - 1) - policy['BULAN_MULAI'] + 1
                    if bulan_terlewat > 0:
                        proyeksi[mask_policy] += (tambahan * bulan_terlewat)

        df_fc[f'FC_{bln}'] = proyeksi
        
    return df_fc

def deteksi_early_warning(df_fc: pd.DataFrame, bulan_penuh_terakhir: int) -> pd.DataFrame:
    """Menganalisis anomali dan risiko penyerapan (Over-budget, Penyerapan Terlalu Cepat/Lambat)"""
    df_ew = df_fc.copy()
    bulan_sisa = BULAN_KOLOM[bulan_penuh_terakhir:]
    
    df_ew['TOTAL_FORECAST'] = df_ew['REALISASI_YTD'] + df_ew[[f'FC_{b}' for b in bulan_sisa]].sum(axis=1)
    df_ew['PERSEN_SERAPAN_AKHIR'] = np.where(df_ew['PAGU'] > 0, df_ew['TOTAL_FORECAST'] / df_ew['PAGU'], 0)
    
    # Inisialisasi Resiko
    df_ew['RISK_SCORE'] = 0
    df_ew['WARNING_MSG'] = ""

    # Rule 1: Over Pagu (Kritis untuk selain 51)
    mask_over = (df_ew['TOTAL_FORECAST'] > df_ew['PAGU']) & (df_ew['JENIS BELANJA'] != 51)
    df_ew.loc[mask_over, 'RISK_SCORE'] += 40
    df_ew.loc[mask_over, 'WARNING_MSG'] += "⚠️ Proyeksi melebih pagu. "
    
    # Rule 2: Anomali Serapan YTD vs Pola Historis
    if bulan_penuh_terakhir > 0 and bulan_penuh_terakhir < 12:
        target_historis_ytd = df_ew[[f'PROF_{b}' for b in BULAN_KOLOM[:bulan_penuh_terakhir]]].sum(axis=1)
        actual_ytd_pct = np.where(df_ew['PAGU'] > 0, df_ew['REALISASI_YTD'] / df_ew['PAGU'], 0)
        
        deviasi = actual_ytd_pct - target_historis_ytd
        
        mask_lambat = deviasi < -0.15 # 15% lebih lambat dari historis
        df_ew.loc[mask_lambat, 'RISK_SCORE'] += 20
        df_ew.loc[mask_lambat, 'WARNING_MSG'] += "📉 Serapan YTD terlalu lambat vs Historis. "
        
        mask_cepat = deviasi > 0.15 # 15% lebih cepat dari historis
        df_ew.loc[mask_cepat, 'RISK_SCORE'] += 15
        df_ew.loc[mask_cepat, 'WARNING_MSG'] += "📈 Serapan YTD terlalu cepat vs Historis. "

    # Kategorikan Risiko
    kondisi = [
        df_ew['RISK_SCORE'] >= 40,
        df_ew['RISK_SCORE'] >= 20,
        df_ew['RISK_SCORE'] < 20
    ]
    pilihan = ['TINGGI', 'SEDANG', 'RENDAH']
    df_ew['KATEGORI_RISIKO'] = np.select(kondisi, pilihan, default='RENDAH')
    
    return df_ew

# --------------------------------------------------------------------------
# VISUALISASI BERBASIS PLOTLY
# --------------------------------------------------------------------------

def plot_aktual_vs_forecast(df_ew: pd.DataFrame, tingkat_agregasi: str = 'JENIS BELANJA'):
    """Grafik garis membandingkan Aktual vs Forecast bulanan gabungan"""
    df_agg = df_ew.groupby(tingkat_agregasi).sum(numeric_only=True).reset_index()
    
    fig = go.Figure()
    for _, row in df_agg.iterrows():
        aktual = [row.get(b, 0) for b in BULAN_KOLOM]
        forecast = [row.get(f'FC_{b}', 0) for b in BULAN_KOLOM]
        
        # Gabungkan untuk garis mulus: aktual jika ada, else forecast
        gabungan = [a if a > 0 else f for a, f in zip(aktual, forecast)]
        
        fig.add_trace(go.Scatter(x=BULAN_KOLOM, y=gabungan, mode='lines+markers', name=str(row[tingkat_agregasi])))
        
    fig.update_layout(title="Proyeksi Bulanan (Aktual & Forecast)", yaxis_title="Rupiah", xaxis_title="Bulan", hovermode="x unified")
    return fig

def plot_heatmap_deviasi(df_ew: pd.DataFrame):
    """Heatmap deviasi persentase serapan proyeksi akhir tahun vs historis per Satker & Jenis Belanja"""
    df_agg = df_ew.groupby(['NMSATKER', 'JENIS BELANJA']).agg(
        PAGU=('PAGU', 'sum'),
        FC=('TOTAL_FORECAST', 'sum')
    ).reset_index()
    
    df_agg['PERSEN_DEV'] = (df_agg['FC'] / (df_agg['PAGU'] + 1e-9) * 100) - 100
    
    pivot = df_agg.pivot(index='NMSATKER', columns='JENIS BELANJA', values='PERSEN_DEV').fillna(0)
    
    fig = px.imshow(pivot, text_auto=".1f", aspect="auto", color_continuous_scale='RdBu_r', 
                    title="Heatmap Deviasi Proyeksi vs Pagu (%) (Merah = Over Budget, Biru = Under Budget)")
    return fig

def plot_waterfall_anggaran(df_ew: pd.DataFrame, level_nama: str):
    """Waterfall chart menunjukkan Pagu awal, Realisasi YTD, dan Sisa Forecast"""
    d_agg = df_ew.sum(numeric_only=True)
    pagu = d_agg['PAGU']
    ytd = d_agg['REALISASI_YTD']
    forecast = d_agg['TOTAL_FORECAST'] - ytd
    deviasi = d_agg['TOTAL_FORECAST'] - pagu

    fig = go.Figure(go.Waterfall(
        name="Anggaran", orientation="v",
        measure=["absolute", "relative", "relative", "total"],
        x=["Pagu Awal", "Realisasi YTD", "Sisa Forecast", "Proyeksi Akhir Tahun"],
        textposition="outside",
        text=[f"Rp {x:,.0f}" for x in [pagu, -ytd, -forecast, pagu + deviasi]],
        y=[pagu, -ytd, -forecast, pagu + deviasi],
        connector={"line":{"color":"rgb(63, 63, 63)"}},
    ))
    fig.update_layout(title=f"Waterfall Anggaran - {level_nama}", showlegend=False)
    return fig