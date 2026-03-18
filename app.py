import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import streamlit as st
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns

st.set_page_config(
    page_title="Battery SoH Analytics",
    page_icon="🔋",
    layout="wide",
    initial_sidebar_state="collapsed"
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Mono:wght@400;500&family=DM+Sans:ital,wght@0,300;0,400;0,500;0,600;1,300&display=swap');

.stApp {
    background: #0d0e11;
    color: #e2e4e9;
    font-family: 'DM Sans', sans-serif;
}
#MainMenu, footer, header { visibility: hidden; }
h1, h2, h3, h4, h5, h6 {
    font-family: 'DM Sans', sans-serif !important;
    letter-spacing: -0.02em !important;
}

/* ── HERO ── */
.hero {
    background: linear-gradient(135deg, #0d0e11 0%, #131520 60%, #0f1019 100%);
    border-bottom: 1px solid #1e2130;
    padding: 2.5rem 2rem 2rem;
    margin-bottom: 2rem;
}
.hero-eyebrow {
    font-family: 'DM Mono', monospace;
    font-size: 0.68rem;
    color: #4f8ef7;
    letter-spacing: 0.2em;
    text-transform: uppercase;
    margin-bottom: 10px;
}
.hero-title {
    font-size: 2.2rem;
    font-weight: 600;
    color: #f0f2f7;
    letter-spacing: -0.04em;
    margin: 0 0 0.4rem;
    line-height: 1.1;
}
.hero-title span { color: #4f8ef7; }
.hero-subtitle {
    font-size: 0.85rem;
    font-weight: 400;
    color: #6b7280;
    font-family: 'DM Mono', monospace;
    letter-spacing: 0.05em;
    text-transform: uppercase;
    margin-bottom: 1rem;
}
.hero-pills { display: flex; gap: 8px; flex-wrap: wrap; }
.hpill {
    font-family: 'DM Mono', monospace;
    font-size: 0.68rem;
    padding: 3px 10px;
    border-radius: 20px;
    letter-spacing: 0.08em;
}
.hpill-blue  { background: #1a2340; border: 1px solid #2a3a5e; color: #5b8def; }
.hpill-green { background: #0c2a1f; border: 1px solid #14532d; color: #4ade80; }
.hpill-gray  { background: #161a22; border: 1px solid #2a3040; color: #6b7280; }

/* Section card */
.section-card {
    background: #131520;
    border: 1px solid #1e2130;
    border-radius: 12px;
    padding: 1.5rem 1.8rem;
    margin-bottom: 1.5rem;
}
.section-label {
    font-family: 'DM Mono', monospace;
    font-size: 0.68rem;
    color: #4a5066;
    text-transform: uppercase;
    letter-spacing: 0.12em;
    margin-bottom: 0.3rem;
}
.section-title {
    font-size: 1.05rem;
    font-weight: 500;
    color: #c8ccd6;
    margin: 0 0 1.2rem 0;
    letter-spacing: -0.02em;
}

/* Metric cards */
.metric-row { display: flex; gap: 1rem; margin-bottom: 1.5rem; }
.metric-card {
    flex: 1;
    background: #0d0e11;
    border: 1px solid #1e2130;
    border-radius: 10px;
    padding: 1.1rem 1.3rem;
}
.metric-label {
    font-family: 'DM Mono', monospace;
    font-size: 0.65rem;
    color: #4a5066;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    margin-bottom: 0.5rem;
}
.metric-value {
    font-size: 1.7rem;
    font-weight: 600;
    letter-spacing: -0.04em;
    line-height: 1;
    margin-bottom: 0.2rem;
}
.metric-unit { font-size: 0.75rem; color: #6b7280; font-family: 'DM Mono', monospace; }
.metric-good  { color: #34d399; }
.metric-ok    { color: #fbbf24; }
.metric-warn  { color: #f97316; }
.metric-bad   { color: #f87171; }

/* Status banner */
.status-banner {
    padding: 0.75rem 1.2rem;
    border-radius: 8px;
    font-size: 0.85rem;
    font-weight: 500;
    margin-bottom: 1.2rem;
    display: flex;
    align-items: center;
    gap: 0.6rem;
}
.status-good  { background: #0c2a1f; border: 1px solid #14532d; color: #4ade80; }
.status-ok    { background: #1c1800; border: 1px solid #713f12; color: #fde68a; }
.status-warn  { background: #1f1100; border: 1px solid #7c2d12; color: #fb923c; }
.status-bad   { background: #1f0a0a; border: 1px solid #7f1d1d; color: #fca5a5; }

/* Interpretation box */
.interpret-box {
    background: #0d0e11;
    border-left: 2px solid #2a3a5e;
    padding: 0.9rem 1.2rem;
    border-radius: 0 8px 8px 0;
    margin-top: 1rem;
    font-size: 0.87rem;
    color: #8892a4;
    line-height: 1.6;
}
.interpret-highlight { color: #5b8def; font-weight: 500; }

/* Upload zone */
.stFileUploader > div {
    background: #131520 !important;
    border: 1.5px dashed #2a3040 !important;
    border-radius: 10px !important;
}

/* Download button */
.stDownloadButton > button {
    background: #131520 !important;
    color: #c8ccd6 !important;
    border: 1px solid #2a3040 !important;
    border-radius: 8px !important;
    font-family: 'DM Mono', monospace !important;
    font-size: 0.8rem !important;
    letter-spacing: 0.05em !important;
    padding: 0.5rem 1.2rem !important;
    transition: all 0.2s !important;
}
.stDownloadButton > button:hover {
    border-color: #5b8def !important;
    color: #5b8def !important;
}
</style>
""", unsafe_allow_html=True)

# ── Matplotlib style ──────────────────────────────────────────────────────────
DARK_BG = "#0d0e11"
CARD_BG = "#131520"
BORDER  = "#1e2130"
TEXT_PRI = "#c8ccd6"
TEXT_MUT = "#4a5066"
BLUE    = "#5b8def"
GREEN   = "#34d399"
ORANGE  = "#f97316"
RED_S   = "#f87171"
GOLD    = "#fbbf24"
GRID_C  = "#1a1d28"

def apply_style(fig, ax_list=None):
    fig.patch.set_facecolor(CARD_BG)
    for ax in (ax_list or fig.get_axes()):
        ax.set_facecolor(DARK_BG)
        ax.spines[:].set_color(BORDER)
        ax.tick_params(colors=TEXT_MUT, labelsize=9)
        ax.xaxis.label.set_color(TEXT_MUT)
        ax.yaxis.label.set_color(TEXT_MUT)
        ax.title.set_color(TEXT_PRI)
        ax.grid(color=GRID_C, linewidth=0.5, linestyle="--", alpha=0.7)
    return fig

# ── Modele ────────────────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("models/lstm_soh_model.h5", compile=False)

model = load_model()

# ── HERO ─────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
  <div class="hero-eyebrow">Analyse predictive de batterie</div>
  <div class="hero-title">Battery <span>State of Health</span> Analytics</div>
  <div class="hero-subtitle">Reseau neuronal LSTM — Diagnostic de degradation</div>
  <div class="hero-pills">
    <span class="hpill hpill-green">LSTM ACTIF</span>
    <span class="hpill hpill-blue">WINDOW = 5</span>
    <span class="hpill hpill-gray">MINMAXSCALER</span>
    <span class="hpill hpill-gray">MODEL lstm_soh_v1.h5</span>
  </div>
</div>
""", unsafe_allow_html=True)

# ── Upload ────────────────────────────────────────────────────────────────────
col_up, col_hint = st.columns([2, 1])
with col_up:
    st.markdown('<div class="section-label">Donnees d\'entree</div>', unsafe_allow_html=True)
    uploaded_file = st.file_uploader(
        "Importer un fichier de mesures (CSV ou Excel)",
        type=["csv", "xlsx"],
        label_visibility="collapsed"
    )
with col_hint:
    st.markdown("""
    <div style="padding:1rem;background:#0d0e11;border:1px solid #1e2130;border-radius:10px;margin-top:1.5rem">
      <div class="section-label">Colonnes requises</div>
      <div style="font-family:'DM Mono',monospace;font-size:0.75rem;color:#4a5066;line-height:2">
        Voltage_measured<br>Current_measured<br>Temperature_measured<br>SoC<br>cycle_number
      </div>
    </div>
    """, unsafe_allow_html=True)

# ── Analyse ───────────────────────────────────────────────────────────────────
if uploaded_file is not None:
    if uploaded_file.name.endswith(".csv"):
        df = pd.read_csv(uploaded_file)
    else:
        df = pd.read_excel(uploaded_file)

    required_columns = [
        "Voltage_measured", "Current_measured",
        "Temperature_measured", "SoC", "cycle_number"
    ]

    if not all(col in df.columns for col in required_columns):
        st.markdown("""
        <div class="status-banner status-bad">
            Colonnes manquantes — verifiez que votre fichier contient les 5 champs requis.
        </div>
        """, unsafe_allow_html=True)
        st.stop()

    # ── Apercu ────────────────────────────────────────────────────────────────
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-label">Inspection</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Apercu du jeu de donnees</div>', unsafe_allow_html=True)
    st.dataframe(df.head(8), use_container_width=True, hide_index=True)
    st.markdown(f'<div style="font-family:\'DM Mono\',monospace;font-size:0.72rem;color:#4a5066;margin-top:0.5rem">{len(df):,} enregistrements &nbsp;·&nbsp; {len(df.columns)} variables</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

    # ── Prediction ────────────────────────────────────────────────────────────
    from sklearn.preprocessing import MinMaxScaler
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(df[required_columns])

    def create_sequences(data, window_size=5):
        return np.array([data[i:i+window_size] for i in range(len(data) - window_size)])

    X_seq = create_sequences(X_scaled)
    predictions = model.predict(X_seq, verbose=0)
    df_result = df.iloc[5:].copy()
    df_result["Predicted_SoH"] = predictions

    avg_soh = df_result["Predicted_SoH"].mean()
    min_soh = df_result["Predicted_SoH"].min()
    max_soh = df_result["Predicted_SoH"].max()
    trend   = df_result["Predicted_SoH"].iloc[-1] - df_result["Predicted_SoH"].iloc[0]

    def soh_color_class(v):
        if v > 80: return "metric-good"
        if v > 60: return "metric-ok"
        if v > 40: return "metric-warn"
        return "metric-bad"

    def soh_status(v):
        if v > 80: return "status-good", "Batterie en bon etat — degradation faible detectee."
        if v > 60: return "status-ok",   "Batterie correcte — degradation progressive a surveiller."
        if v > 40: return "status-warn", "Batterie en degradation avancee — remplacement recommande."
        return "status-bad", "Batterie fortement degradee — remplacement urgent."

    status_cls, status_msg = soh_status(avg_soh)
    trend_str = f"{'−' if trend < 0 else '+'}{abs(trend):.1f}%"

    # ── Metriques ─────────────────────────────────────────────────────────────
    st.markdown(f"""
    <div class="metric-row">
      <div class="metric-card">
        <div class="metric-label">SoH moyen predit</div>
        <div class="metric-value {soh_color_class(avg_soh)}">{avg_soh:.1f}<span style="font-size:1rem">%</span></div>
        <div class="metric-unit">State of Health</div>
      </div>
      <div class="metric-card">
        <div class="metric-label">Minimum observe</div>
        <div class="metric-value metric-bad">{min_soh:.1f}<span style="font-size:1rem">%</span></div>
        <div class="metric-unit">SoH minimal</div>
      </div>
      <div class="metric-card">
        <div class="metric-label">Maximum observe</div>
        <div class="metric-value metric-good">{max_soh:.1f}<span style="font-size:1rem">%</span></div>
        <div class="metric-unit">SoH maximal</div>
      </div>
      <div class="metric-card">
        <div class="metric-label">Tendance nette</div>
        <div class="metric-value {'metric-bad' if trend < 0 else 'metric-good'}">{trend_str}</div>
        <div class="metric-unit">Variation debut vers fin</div>
      </div>
    </div>
    <div class="status-banner {status_cls}">{status_msg}</div>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    # ── 1. Distribution SoH ───────────────────────────────────────────────────
    if "SoH" in df.columns:
        with col1:
            st.markdown('<div class="section-card">', unsafe_allow_html=True)
            st.markdown('<div class="section-label">Statistiques</div>', unsafe_allow_html=True)
            st.markdown('<div class="section-title">Distribution du SoH reel</div>', unsafe_allow_html=True)

            fig1, ax1 = plt.subplots(figsize=(5.5, 3.2))
            ax1.hist(df["SoH"], bins=30, color=BLUE, alpha=0.85, edgecolor=DARK_BG, linewidth=0.5)
            ax1.axvline(df["SoH"].mean(), color=GOLD, linewidth=1.2, linestyle="--", alpha=0.9, label=f"Moy. {df['SoH'].mean():.1f}%")
            ax1.set_xlabel("SoH (%)", fontsize=9)
            ax1.set_ylabel("Frequence", fontsize=9)
            ax1.legend(fontsize=8, framealpha=0, labelcolor=GOLD)
            apply_style(fig1)
            fig1.tight_layout(pad=0.8)
            st.pyplot(fig1)

            mean_soh = df["SoH"].mean()
            if mean_soh > 80:   detail = "La majorite des batteries presentent un etat de sante satisfaisant."
            elif mean_soh > 60: detail = "Une degradation progressive est observable sur l'ensemble de la flotte."
            else:               detail = "La flotte presente une degradation avancee."

            st.markdown(f"""
            <div class="interpret-box">
              Distribution centree sur <span class="interpret-highlight">{mean_soh:.1f}%</span>. {detail}
            </div>
            """, unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)

    # ── 2. Correlation ────────────────────────────────────────────────────────
    with col2:
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.markdown('<div class="section-label">Analyse</div>', unsafe_allow_html=True)
        st.markdown('<div class="section-title">Matrice de correlation</div>', unsafe_allow_html=True)

        fig2, ax2 = plt.subplots(figsize=(5.5, 3.2))
        corr = df.corr(numeric_only=True)
        cmap = sns.diverging_palette(230, 20, as_cmap=True)
        sns.heatmap(
            corr, ax=ax2, cmap=cmap, annot=True, fmt=".2f",
            linewidths=0.4, linecolor=DARK_BG,
            annot_kws={"size": 7, "color": TEXT_PRI},
            cbar_kws={"shrink": 0.75}
        )
        ax2.tick_params(labelsize=7.5, colors=TEXT_MUT, rotation=30)
        for _, spine in ax2.spines.items():
            spine.set_visible(False)
        ax2.set_facecolor(CARD_BG)
        fig2.patch.set_facecolor(CARD_BG)
        cbar = ax2.collections[0].colorbar
        cbar.ax.tick_params(labelsize=7, colors=TEXT_MUT)
        fig2.tight_layout(pad=0.8)
        st.pyplot(fig2)

        if "SoH" in corr.columns:
            top_var = corr["SoH"].drop("SoH").abs().sort_values(ascending=False).index[0]
            top_val = corr["SoH"][top_var]
            st.markdown(f"""
            <div class="interpret-box">
              Variable la plus correlee au SoH : <span class="interpret-highlight">{top_var}</span>
              (r = {top_val:.2f}). Les valeurs proches de ±1 indiquent une relation lineaire forte.
            </div>
            """, unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    # ── 3. Evolution SoH ──────────────────────────────────────────────────────
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-label">Prediction LSTM</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Evolution du SoH predit par cycles</div>', unsafe_allow_html=True)

    fig3, ax3 = plt.subplots(figsize=(11, 3.5))
    x = np.arange(len(df_result))
    ax3.fill_between(x, df_result["Predicted_SoH"].values, alpha=0.12, color=BLUE)
    ax3.plot(x, df_result["Predicted_SoH"].values, color=BLUE, linewidth=1.5, label="SoH predit")
    if "SoH" in df_result.columns:
        ax3.plot(x, df_result["SoH"].values, color=TEXT_MUT, linewidth=0.8, linestyle="--", alpha=0.6, label="SoH reel")
    ax3.axhline(80, color=GREEN, linewidth=0.6, linestyle=":", alpha=0.4)
    ax3.axhline(60, color=GOLD,  linewidth=0.6, linestyle=":", alpha=0.4)
    ax3.set_xlabel("Index de cycle", fontsize=9)
    ax3.set_ylabel("SoH (%)", fontsize=9)
    ax3.legend(fontsize=8, framealpha=0, labelcolor=TEXT_PRI)
    apply_style(fig3)
    fig3.tight_layout(pad=0.8)
    st.pyplot(fig3)

    if trend < 0:
        evol_msg = f"Tendance decroissante de <span class='interpret-highlight'>{abs(trend):.1f}%</span> sur l'ensemble des cycles, confirmant la degradation progressive."
    else:
        evol_msg = "Le SoH reste stable sur la periode d'observation."
    st.markdown(f'<div class="interpret-box">{evol_msg}</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

    # ── 4. Reel vs Predit ─────────────────────────────────────────────────────
    if "SoH" in df.columns:
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.markdown('<div class="section-label">Evaluation</div>', unsafe_allow_html=True)
        st.markdown('<div class="section-title">Valeurs reelles vs predites</div>', unsafe_allow_html=True)

        y_true = df_result["SoH"].values[:100]
        y_pred = df_result["Predicted_SoH"].values[:100]

        fig4, ax4 = plt.subplots()
        ax4.scatter(y_true, y_pred, alpha=0.65, color=BLUE, s=18, edgecolors="none")
        min_val = min(min(y_true), min(y_pred))
        max_val = max(max(y_true), max(y_pred))
        ax4.plot([min_val, max_val], [min_val, max_val], color=TEXT_MUT, linewidth=1, linestyle="--", alpha=0.5)
        ax4.set_xlabel("SoH reel (%)", fontsize=9)
        ax4.set_ylabel("SoH predit (%)", fontsize=9)
        mae = np.mean(np.abs(y_true - y_pred))
        ax4.text(min_val + 0.5, max_val - 1.5, f"MAE: {mae:.2f}%", fontsize=9, color=TEXT_PRI,
                 bbox=dict(facecolor=CARD_BG, edgecolor=BORDER, alpha=0.9, boxstyle='round,pad=0.4'))
        apply_style(fig4)
        fig4.tight_layout(pad=0.8)
        st.pyplot(fig4)

        if mae < 5:   qual = "Les points sont proches de la diagonale — modele precis."
        elif mae < 10: qual = "Le modele est correct avec quelques ecarts."
        else:          qual = "Les predictions presentent des ecarts importants."
        st.markdown(f"""
        <div class="interpret-box">
          Comparaison sur les 100 premiers cycles. La diagonale represente une prediction parfaite.
          {qual}
        </div>
        """, unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    # ── 5. Resultats ──────────────────────────────────────────────────────────
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-label">Resultats</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Tableau des predictions</div>', unsafe_allow_html=True)
    st.dataframe(df_result.head(20), use_container_width=True, hide_index=True)
    st.markdown('</div>', unsafe_allow_html=True)

    # ── 6. Global ─────────────────────────────────────────────────────────────
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-label">Diagnostic</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Interpretation globale et causes</div>', unsafe_allow_html=True)
    st.markdown(f'<div style="font-family:\'DM Mono\',monospace;font-size:0.85rem;color:#c8ccd6;margin-bottom:1rem">SoH moyen : <span style="color:{("#34d399" if avg_soh>80 else "#fbbf24" if avg_soh>60 else "#f97316" if avg_soh>40 else "#f87171")}">{avg_soh:.2f}%</span></div>', unsafe_allow_html=True)

    causes = []
    if df_result["cycle_number"].mean() > 500:
        causes.append("Nombre eleve de cycles — vieillissement naturel des electrodes.")
    if df_result["Temperature_measured"].mean() > 35:
        causes.append(f"Temperature elevee ({df_result['Temperature_measured'].mean():.1f}°C) — degradation acceleree.")
    if df_result["SoC"].mean() < 30:
        causes.append(f"SoC faible ({df_result['SoC'].mean():.1f}%) — stress electrochimique par decharge profonde.")

    if not causes:
        st.markdown('<div class="interpret-box">Aucune cause critique detectee dans les seuils definis.</div>', unsafe_allow_html=True)
    else:
        for c in causes:
            st.markdown(f'<div class="interpret-box" style="margin-bottom:0.5rem">— {c}</div>', unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)

    # ── Export ────────────────────────────────────────────────────────────────
    csv = df_result.to_csv(index=False).encode('utf-8')
    st.download_button("Telecharger les resultats (CSV)", csv, "resultats_soh.csv", mime="text/csv")

    # ── Footer ────────────────────────────────────────────────────────────────
    st.markdown("""
    <div style="text-align:center;padding:2rem 0 1rem;font-family:'DM Mono',monospace;
                font-size:0.65rem;color:#2a3040;letter-spacing:0.1em">
      BATTERY SOH ANALYTICS · LSTM PREDICTIVE MODEL
    </div>
    """, unsafe_allow_html=True)