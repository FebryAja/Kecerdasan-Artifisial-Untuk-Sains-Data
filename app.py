import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import io
import warnings
warnings.filterwarnings('ignore')

from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Prediksi Panen Sawit",
    page_icon="🌴",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─────────────────────────────────────────────
# CUSTOM CSS
# ─────────────────────────────────────────────
st.markdown("""
<style>
    .main { background-color: #f8fdf8; }
    .block-container { padding-top: 1.5rem; }
    .metric-card {
        background: white;
        border-radius: 12px;
        padding: 18px 20px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.08);
        border-left: 5px solid #27ae60;
        margin-bottom: 10px;
    }
    .result-box {
        border-radius: 14px;
        padding: 22px;
        text-align: center;
        color: white;
        margin: 12px 0;
    }
    .stTabs [data-baseweb="tab-list"] { gap: 8px; }
    .stTabs [data-baseweb="tab"] {
        border-radius: 8px 8px 0 0;
        padding: 8px 20px;
        font-weight: 600;
    }
    h1 { color: #1a5c2e !important; }
    h2, h3 { color: #2c7a44 !important; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# TRAIN MODEL (cached)
# ─────────────────────────────────────────────
@st.cache_resource(show_spinner="⏳ Melatih model ML...")
def train_models():
    np.random.seed(42)
    n = 1200

    umur         = np.random.uniform(3, 25, n)
    luas         = np.random.uniform(0.5, 100, n)
    hujan        = np.random.uniform(100, 400, n)
    suhu         = np.random.uniform(24, 34, n)
    pupuk        = np.random.uniform(50, 500, n)
    tk           = np.random.uniform(0.5, 5, n)
    tanah_cat    = np.random.choice(['Gambut','Mineral','Latosol','Podsolik'], n)
    varietas_cat = np.random.choice(['Tenera','Dura','Pisifera','DxP Unggul'], n)

    tanah_map    = {'Gambut':0.85,'Mineral':1.0,'Latosol':1.1,'Podsolik':0.9}
    varietas_map = {'Tenera':1.0,'Dura':0.85,'Pisifera':0.75,'DxP Unggul':1.15}
    tf = np.array([tanah_map[t] for t in tanah_cat])
    vf = np.array([varietas_map[v] for v in varietas_cat])

    fase = np.where(umur<3, 0,
           np.where(umur<=8, (umur-3)/5,
           np.where(umur<=15, 1.0,
           np.where(umur<=25, 1.0-(umur-15)*0.04, 0.5))))

    curah_f = np.clip((hujan-100)/300, 0.3, 1.0)
    suhu_f  = np.where((suhu>=26)&(suhu<=32), 1.0, np.where(suhu<26, 0.85, 0.9))
    pupuk_f = np.clip(pupuk/250, 0.5, 1.2)
    tk_f    = np.clip(tk/2, 0.7, 1.1)

    hasil = np.clip(20*fase*curah_f*suhu_f*pupuk_f*tk_f*tf*vf + np.random.normal(0,1,n), 0, 35)

    le_tanah    = LabelEncoder().fit(tanah_cat)
    le_varietas = LabelEncoder().fit(varietas_cat)

    df = pd.DataFrame({
        'Umur_Tanaman':umur,'Luas_Lahan':luas,'Curah_Hujan':hujan,
        'Suhu':suhu,'Jumlah_Pupuk':pupuk,'Tenaga_Kerja':tk,
        'Jenis_Tanah':le_tanah.transform(tanah_cat),
        'Varietas':le_varietas.transform(varietas_cat),
        'Hasil_Panen':hasil
    })

    feats = ['Umur_Tanaman','Luas_Lahan','Curah_Hujan','Suhu','Jumlah_Pupuk','Tenaga_Kerja','Jenis_Tanah','Varietas']
    X, y  = df[feats], df['Hasil_Panen']
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    Xtr_s  = scaler.fit_transform(Xtr)
    Xte_s  = scaler.transform(Xte)

    models_dict = {
        'Random Forest':     RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1),
        'Gradient Boosting': GradientBoostingRegressor(n_estimators=200, random_state=42),
        'Ridge Regression':  Ridge(alpha=1.0)
    }
    hasil_model = {}
    for nm, mdl in models_dict.items():
        if nm == 'Ridge Regression':
            mdl.fit(Xtr_s, ytr); yp = mdl.predict(Xte_s)
        else:
            mdl.fit(Xtr, ytr); yp = mdl.predict(Xte)
        hasil_model[nm] = {
            'model': mdl,
            'MAE':  round(mean_absolute_error(yte, yp), 3),
            'RMSE': round(np.sqrt(mean_squared_error(yte, yp)), 3),
            'R2':   round(r2_score(yte, yp), 4),
            'y_pred': yp, 'y_test': yte
        }

    return hasil_model, scaler, le_tanah, le_varietas, feats, df


# ─────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────
def predict(vals, hasil_model, scaler, le_tanah, le_varietas, feats):
    tanah_e    = le_tanah.transform([vals['Jenis_Tanah']])[0]
    varietas_e = le_varietas.transform([vals['Varietas']])[0]
    X = np.array([[vals['Umur_Tanaman'], vals['Luas_Lahan'], vals['Curah_Hujan'],
                   vals['Suhu'], vals['Jumlah_Pupuk'], vals['Tenaga_Kerja'],
                   tanah_e, varietas_e]])
    Xs = scaler.transform(X)

    preds = {}
    for nm, info in hasil_model.items():
        inp = Xs if nm == 'Ridge Regression' else X
        preds[nm] = max(0, info['model'].predict(inp)[0])

    r2s   = [hasil_model[m]['R2'] for m in preds]
    total = sum(r2s)
    bobot = [r/total for r in r2s]
    ens   = sum(b*p for b, p in zip(bobot, preds.values()))
    preds['Ensemble'] = max(0, ens)
    return preds


def kategori(val):
    if val >= 22:   return "SANGAT TINGGI", "#27ae60"
    elif val >= 18: return "TINGGI",         "#f39c12"
    elif val >= 13: return "SEDANG",          "#e67e22"
    else:           return "RENDAH",          "#e74c3c"


def rekomendasi(umur, hujan, suhu, pupuk, tk):
    s = []
    if umur < 3:    s.append("⚠️ Tanaman belum fase produksi (minimal 3 tahun).")
    elif umur > 20: s.append("⚠️ Tanaman memasuki fase tua — pertimbangkan peremajaan.")
    if hujan < 150: s.append("💧 Curah hujan kurang — pertimbangkan irigasi tambahan.")
    elif hujan > 350: s.append("🌧️ Curah hujan berlebih — pastikan drainase lahan baik.")
    if suhu < 26:   s.append("🌡️ Suhu terlalu rendah — berpotensi menghambat pertumbuhan.")
    elif suhu > 32: s.append("🔥 Suhu terlalu tinggi — waspadai stres tanaman.")
    if pupuk < 150: s.append("🌱 Dosis pupuk sangat rendah — tingkatkan pemupukan.")
    elif pupuk < 250: s.append("🌱 Pertimbangkan meningkatkan dosis pupuk.")
    if tk < 1.0:    s.append("👷 Tenaga kerja kurang — panen bisa terlambat.")
    if not s:       s.append("✅ Kondisi lahan sudah optimal! Pertahankan praktik budidaya.")
    return s


# ─────────────────────────────────────────────
# LOAD MODEL
# ─────────────────────────────────────────────
hasil_model, scaler, le_tanah, le_varietas, feats, df_data = train_models()

# ─────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────
st.markdown("""
<div style="background:linear-gradient(135deg,#1a5c2e,#27ae60);padding:28px 32px;border-radius:16px;margin-bottom:24px;">
  <h1 style="color:white!important;margin:0;font-size:2rem;">🌴 Prediksi Hasil Panen Kelapa Sawit</h1>
  <p style="color:#d5f5e3;margin:8px 0 0;font-size:1rem;">Estimasi produksi TBS (Tandan Buah Segar) menggunakan Machine Learning</p>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# TABS
# ─────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs(["🔍 Prediksi Manual", "📤 Upload CSV", "📊 Performa Model", "📈 Analisis Sensitivitas"])


# ══════════════════════════════════════════════
# TAB 1 — PREDIKSI MANUAL
# ══════════════════════════════════════════════
with tab1:
    st.subheader("Masukkan Data Lahan Anda")

    col_a, col_b = st.columns(2)

    with col_a:
        st.markdown("**📋 Data Tanaman & Lahan**")
        umur  = st.slider("Umur Tanaman (tahun)", 1.0, 25.0, 8.0, 0.5)
        luas  = st.number_input("Luas Lahan (ha)", min_value=0.1, max_value=500.0, value=10.0, step=0.5)
        tanah = st.selectbox("Jenis Tanah", ['Mineral','Latosol','Gambut','Podsolik'])
        var   = st.selectbox("Varietas", ['Tenera','DxP Unggul','Dura','Pisifera'])

    with col_b:
        st.markdown("**🌦️ Data Iklim & Manajemen**")
        hujan = st.slider("Curah Hujan (mm/bulan)", 50, 450, 220, 5)
        suhu  = st.slider("Suhu Rata-rata (°C)", 22.0, 38.0, 28.0, 0.5)
        pupuk = st.slider("Jumlah Pupuk (kg/ha)", 50, 600, 250, 10)
        tk    = st.slider("Tenaga Kerja (orang/ha)", 0.5, 6.0, 2.0, 0.1)

    st.markdown("---")
    btn = st.button("🌴 PREDIKSI HASIL PANEN", type="primary", use_container_width=True)

    if btn:
        vals = {'Umur_Tanaman':umur,'Luas_Lahan':luas,'Curah_Hujan':hujan,
                'Suhu':suhu,'Jumlah_Pupuk':pupuk,'Tenaga_Kerja':tk,
                'Jenis_Tanah':tanah,'Varietas':var}

        preds = predict(vals, hasil_model, scaler, le_tanah, le_varietas, feats)
        ens   = preds['Ensemble']
        kat, warna = kategori(ens)
        total_prod = ens * luas
        saran = rekomendasi(umur, hujan, suhu, pupuk, tk)

        # Hasil utama
        st.markdown(f"""
        <div class="result-box" style="background:{warna};">
            <div style="font-size:2.8rem;font-weight:800;">{ens:.2f} <span style="font-size:1.4rem;">ton/ha</span></div>
            <div style="font-size:1.2rem;margin-top:4px;">Kategori: <b>{kat}</b></div>
            <div style="font-size:1rem;margin-top:6px;opacity:0.9;">Total Produksi Estimasi: <b>{total_prod:.1f} ton TBS</b> ({luas:.1f} ha)</div>
        </div>
        """, unsafe_allow_html=True)

        # Tabel per model
        c1, c2, c3 = st.columns(3)
        for col, nm, icon in zip([c1,c2,c3],
                                  ['Random Forest','Gradient Boosting','Ridge Regression'],
                                  ['🌳','📈','📐']):
            with col:
                st.markdown(f"""
                <div class="metric-card">
                  <div style="font-size:.85rem;color:#666;">{icon} {nm}</div>
                  <div style="font-size:1.7rem;font-weight:700;color:#1a5c2e;">{preds[nm]:.2f}</div>
                  <div style="font-size:.8rem;color:#888;">ton/ha &nbsp;|&nbsp; R²={hasil_model[nm]['R2']}</div>
                </div>
                """, unsafe_allow_html=True)

        # Bar chart
        fig, ax = plt.subplots(figsize=(8, 3.2))
        labels  = ['Random\nForest','Gradient\nBoosting','Ridge\nRegression','Ensemble']
        values  = [preds['Random Forest'], preds['Gradient Boosting'], preds['Ridge Regression'], ens]
        colors  = ['#2ecc71','#3498db','#e74c3c','#9b59b6']
        bars    = ax.bar(labels, values, color=colors, alpha=0.85, width=0.5, edgecolor='white', linewidth=1.5)
        for bar, val in zip(bars, values):
            ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.15, f'{val:.2f}',
                    ha='center', va='bottom', fontweight='bold', fontsize=11)
        ax.axhline(20, color='gray', linestyle='--', linewidth=1, alpha=0.6, label='Rata-rata Nasional ~20 ton/ha')
        ax.set_ylabel('Prediksi (ton/ha)', fontsize=10)
        ax.set_title('Perbandingan Prediksi Antar Model', fontsize=12, fontweight='bold')
        ax.set_ylim(0, max(values)*1.3+1)
        ax.legend(fontsize=9)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

        # Rekomendasi
        st.markdown("### 💡 Rekomendasi")
        for s in saran:
            st.info(s)


# ══════════════════════════════════════════════
# TAB 2 — UPLOAD CSV
# ══════════════════════════════════════════════
with tab2:
    st.subheader("📤 Prediksi Batch dari File CSV")

    st.markdown("""
    **Format kolom CSV yang diperlukan:**

    | Kolom | Tipe | Nilai yang Valid |
    |---|---|---|
    | `Umur_Tanaman` | Angka | 1–25 (tahun) |
    | `Luas_Lahan` | Angka | > 0 (ha) |
    | `Curah_Hujan` | Angka | mm/bulan |
    | `Suhu` | Angka | °C |
    | `Jumlah_Pupuk` | Angka | kg/ha |
    | `Tenaga_Kerja` | Angka | orang/ha |
    | `Jenis_Tanah` | Teks | Mineral / Gambut / Latosol / Podsolik |
    | `Varietas` | Teks | Tenera / DxP Unggul / Dura / Pisifera |
    """)

    # Tombol download contoh CSV
    contoh = pd.DataFrame({
        'Umur_Tanaman':  [5,10,15,3,22,8,12,6,18,4],
        'Luas_Lahan':    [10,20,8,50,5,15,30,12,25,8],
        'Curah_Hujan':   [200,250,180,300,150,220,270,190,240,310],
        'Suhu':          [28,27,30,26,32,28,27,29,28,27],
        'Jumlah_Pupuk':  [200,350,150,400,100,280,320,230,300,350],
        'Tenaga_Kerja':  [2,3,1.5,4,1,2.5,3,2,2.5,3.5],
        'Jenis_Tanah':   ['Mineral','Latosol','Gambut','Mineral','Podsolik','Mineral','Latosol','Mineral','Latosol','Gambut'],
        'Varietas':      ['Tenera','DxP Unggul','Tenera','Dura','Tenera','DxP Unggul','Tenera','Tenera','DxP Unggul','Tenera']
    })
    csv_bytes = contoh.to_csv(index=False).encode('utf-8')
    st.download_button("⬇️ Download Contoh CSV", data=csv_bytes,
                       file_name="contoh_data_lahan.csv", mime="text/csv")

    uploaded = st.file_uploader("Upload file CSV Anda", type=["csv"])

    if uploaded:
        try:
            df_up = pd.read_csv(uploaded)
            st.success(f"✅ File berhasil diupload — {len(df_up)} baris data")
            st.dataframe(df_up.head(), use_container_width=True)

            with st.spinner("🔄 Memproses prediksi..."):
                df_res = df_up.copy()
                tanah_e    = le_tanah.transform(df_up['Jenis_Tanah'])
                varietas_e = le_varietas.transform(df_up['Varietas'])

                X_in = np.column_stack([
                    df_up['Umur_Tanaman'], df_up['Luas_Lahan'], df_up['Curah_Hujan'],
                    df_up['Suhu'], df_up['Jumlah_Pupuk'], df_up['Tenaga_Kerja'],
                    tanah_e, varietas_e
                ])
                Xs = scaler.transform(X_in)

                p_rf = hasil_model['Random Forest']['model'].predict(X_in)
                p_gb = hasil_model['Gradient Boosting']['model'].predict(X_in)
                p_rd = hasil_model['Ridge Regression']['model'].predict(Xs)

                r2s   = [hasil_model[m]['R2'] for m in ['Random Forest','Gradient Boosting','Ridge Regression']]
                total = sum(r2s)
                bobot = [r/total for r in r2s]
                p_ens = np.maximum(bobot[0]*p_rf + bobot[1]*p_gb + bobot[2]*p_rd, 0)

                df_res['Pred_RF']       = np.round(np.maximum(p_rf, 0), 2)
                df_res['Pred_GB']       = np.round(np.maximum(p_gb, 0), 2)
                df_res['Pred_Ridge']    = np.round(np.maximum(p_rd, 0), 2)
                df_res['Pred_Ensemble'] = np.round(p_ens, 2)
                df_res['Kategori']      = df_res['Pred_Ensemble'].apply(
                    lambda x: 'SANGAT TINGGI' if x>=22 else ('TINGGI' if x>=18 else ('SEDANG' if x>=13 else 'RENDAH'))
                )
                df_res['Total_Produksi_Ton'] = np.round(p_ens * df_up['Luas_Lahan'], 2)

            st.markdown("### 📋 Hasil Prediksi")
            st.dataframe(df_res[['Umur_Tanaman','Luas_Lahan','Pred_RF','Pred_GB',
                                  'Pred_Ridge','Pred_Ensemble','Kategori','Total_Produksi_Ton']],
                         use_container_width=True)

            col_m1, col_m2, col_m3 = st.columns(3)
            col_m1.metric("Rata-rata Panen", f"{df_res['Pred_Ensemble'].mean():.2f} ton/ha")
            col_m2.metric("Total Produksi", f"{df_res['Total_Produksi_Ton'].sum():.1f} ton")
            col_m3.metric("Lahan Optimal (≥18 t/ha)", f"{(df_res['Pred_Ensemble']>=18).sum()} dari {len(df_res)}")

            out_csv = df_res.to_csv(index=False).encode('utf-8')
            st.download_button("⬇️ Download Hasil Prediksi CSV", data=out_csv,
                               file_name="hasil_prediksi_sawit.csv", mime="text/csv")

        except Exception as e:
            st.error(f"❌ Error memproses file: {e}")


# ══════════════════════════════════════════════
# TAB 3 — PERFORMA MODEL
# ══════════════════════════════════════════════
with tab3:
    st.subheader("📊 Performa Model Machine Learning")

    # Tabel metrik
    metrik_df = pd.DataFrame([
        {'Model': nm, 'MAE (ton/ha)': info['MAE'], 'RMSE (ton/ha)': info['RMSE'], 'R²': info['R2']}
        for nm, info in hasil_model.items()
    ])
    best_r2 = metrik_df['R²'].max()
    st.dataframe(metrik_df.style.highlight_max(subset=['R²'], color='#d5f5e3')
                               .highlight_min(subset=['MAE (ton/ha)','RMSE (ton/ha)'], color='#d5f5e3'),
                 use_container_width=True)

    fig2, axes2 = plt.subplots(1, 3, figsize=(16, 5))
    fig2.suptitle('Performa Model — Actual vs Predicted', fontsize=14, fontweight='bold')
    colors2 = ['#2ecc71','#3498db','#e74c3c']

    for idx, (nm, info) in enumerate(hasil_model.items()):
        ax = axes2[idx]
        yp = info['y_pred']
        yt = list(info['y_test'])
        ax.scatter(yt, yp, alpha=0.35, color=colors2[idx], s=15)
        lims = [min(min(yt), min(yp)), max(max(yt), max(yp))]
        ax.plot(lims, lims, 'r--', linewidth=1.5)
        ax.set_xlabel('Aktual (ton/ha)', fontsize=10)
        ax.set_ylabel('Prediksi (ton/ha)', fontsize=10)
        ax.set_title(f'{nm}\nR²={info["R2"]:.4f} | MAE={info["MAE"]:.3f}', fontsize=10, fontweight='bold')

    plt.tight_layout()
    st.pyplot(fig2)
    plt.close()

    # Feature importance
    st.markdown("### 🔎 Feature Importance (Random Forest)")
    rf = hasil_model['Random Forest']['model']
    fi = rf.feature_importances_
    feat_labels = ['Umur Tanaman','Luas Lahan','Curah Hujan','Suhu','Pupuk','T.Kerja','J.Tanah','Varietas']
    idx_sorted  = np.argsort(fi)

    fig3, ax3 = plt.subplots(figsize=(8, 4))
    bars3 = ax3.barh([feat_labels[i] for i in idx_sorted], fi[idx_sorted], color='#27ae60', alpha=0.8)
    ax3.set_xlabel('Importance', fontsize=10)
    ax3.set_title('Pengaruh Fitur terhadap Prediksi Panen', fontsize=11, fontweight='bold')
    for bar in bars3:
        ax3.text(bar.get_width()+0.002, bar.get_y()+bar.get_height()/2,
                 f'{bar.get_width():.3f}', va='center', fontsize=9)
    plt.tight_layout()
    st.pyplot(fig3)
    plt.close()


# ══════════════════════════════════════════════
# TAB 4 — ANALISIS SENSITIVITAS
# ══════════════════════════════════════════════
with tab4:
    st.subheader("📈 Analisis Sensitivitas Faktor")
    st.info("Grafik ini menunjukkan bagaimana perubahan satu faktor memengaruhi prediksi hasil panen, sementara faktor lain dibuat konstan.")

    default_v = {
        'Umur_Tanaman':10,'Luas_Lahan':10,'Curah_Hujan':220,
        'Suhu':28,'Jumlah_Pupuk':300,'Tenaga_Kerja':2.5,
        'Jenis_Tanah': le_tanah.transform(['Mineral'])[0],
        'Varietas':    le_varietas.transform(['Tenera'])[0]
    }

    analisis = {
        'Umur Tanaman (tahun)':    ('Umur_Tanaman',  np.arange(3, 26, 0.5)),
        'Curah Hujan (mm/bulan)':  ('Curah_Hujan',   np.arange(80, 420, 10)),
        'Suhu (°C)':               ('Suhu',           np.arange(22, 38, 0.5)),
        'Jumlah Pupuk (kg/ha)':    ('Jumlah_Pupuk',  np.arange(50, 600, 10)),
        'Tenaga Kerja (orang/ha)': ('Tenaga_Kerja',   np.arange(0.5, 6, 0.1)),
    }

    rf_model = hasil_model['Random Forest']['model']
    pal      = ['#e74c3c','#3498db','#27ae60','#f39c12','#9b59b6']

    fig4, axes4 = plt.subplots(2, 3, figsize=(16, 9))
    fig4.suptitle('Analisis Sensitivitas — Pengaruh Faktor terhadap Hasil Panen', fontsize=14, fontweight='bold')

    ax_flat = axes4.flatten()
    for idx, (label, (kolom, rentang)) in enumerate(analisis.items()):
        preds4 = []
        for val in rentang:
            row = [default_v[f] for f in feats]
            row[feats.index(kolom)] = val
            p = max(0, rf_model.predict(np.array([row]))[0])
            preds4.append(p)
        ax = ax_flat[idx]
        ax.plot(rentang, preds4, color=pal[idx], linewidth=2.5)
        ax.fill_between(rentang, preds4, alpha=0.15, color=pal[idx])
        ax.axhline(np.mean(preds4), color='gray', linestyle='--', linewidth=1, alpha=0.6)
        ax.set_xlabel(label, fontsize=10)
        ax.set_ylabel('Prediksi (ton/ha)', fontsize=10)
        ax.set_title(f'Pengaruh {label}', fontsize=10, fontweight='bold')
        ax.set_ylim(bottom=0)

    ax_flat[-1].axis('off')
    plt.tight_layout()
    st.pyplot(fig4)
    plt.close()

# ─────────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────────
st.markdown("---")
st.markdown("""
<div style="text-align:center;color:#888;font-size:0.85rem;">
  🌴 Aplikasi Prediksi Panen Kelapa Sawit &nbsp;|&nbsp; Model: Random Forest · Gradient Boosting · Ridge Regression<br>
  <i>Disclaimer: Model menggunakan data sintetis. Untuk akurasi optimal, latih ulang dengan data historis lahan Anda.</i>
</div>
""", unsafe_allow_html=True)
