import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report, confusion_matrix,
    roc_auc_score, roc_curve, precision_recall_curve
)
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
import warnings
warnings.filterwarnings("ignore")

# ─── Page Config ────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="FraudSense · Credit Card Fraud Detector",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─── Custom CSS ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=DM+Mono:wght@300;400;500&display=swap');

html, body, [class*="css"] {
    font-family: 'DM Mono', monospace;
    background-color: #0a0e1a;
    color: #e2e8f0;
}

.stApp {
    background: linear-gradient(135deg, #0a0e1a 0%, #0f1829 50%, #0a0e1a 100%);
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background: #0d1220;
    border-right: 1px solid #1e2d4a;
}

/* Hero Title */
.hero-title {
    font-family: 'Syne', sans-serif;
    font-size: 3rem;
    font-weight: 800;
    background: linear-gradient(90deg, #38bdf8, #818cf8, #f472b6);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    letter-spacing: -1px;
    line-height: 1.1;
    margin-bottom: 0.3rem;
}

.hero-sub {
    font-family: 'DM Mono', monospace;
    font-size: 0.85rem;
    color: #64748b;
    letter-spacing: 2px;
    text-transform: uppercase;
    margin-bottom: 2rem;
}

/* Metric Cards */
.metric-card {
    background: linear-gradient(145deg, #111827, #1a2540);
    border: 1px solid #1e3a5f;
    border-radius: 12px;
    padding: 1.4rem 1.6rem;
    text-align: center;
    transition: border-color 0.3s;
}

.metric-card:hover { border-color: #38bdf8; }

.metric-value {
    font-family: 'Syne', sans-serif;
    font-size: 2.2rem;
    font-weight: 800;
    color: #38bdf8;
}

.metric-label {
    font-size: 0.7rem;
    color: #64748b;
    letter-spacing: 2px;
    text-transform: uppercase;
    margin-top: 0.2rem;
}

/* Section Headers */
.section-header {
    font-family: 'Syne', sans-serif;
    font-size: 1.2rem;
    font-weight: 700;
    color: #e2e8f0;
    border-left: 3px solid #38bdf8;
    padding-left: 0.8rem;
    margin: 1.5rem 0 1rem 0;
}

/* Alert Boxes */
.fraud-alert {
    background: linear-gradient(135deg, #450a0a, #7f1d1d);
    border: 1px solid #ef4444;
    border-radius: 10px;
    padding: 1.2rem 1.5rem;
    margin: 1rem 0;
    font-family: 'Syne', sans-serif;
    font-size: 1.1rem;
    font-weight: 600;
    color: #fca5a5;
    text-align: center;
}

.safe-alert {
    background: linear-gradient(135deg, #052e16, #14532d);
    border: 1px solid #22c55e;
    border-radius: 10px;
    padding: 1.2rem 1.5rem;
    margin: 1rem 0;
    font-family: 'Syne', sans-serif;
    font-size: 1.1rem;
    font-weight: 600;
    color: #86efac;
    text-align: center;
}

/* Buttons */
.stButton > button {
    background: linear-gradient(135deg, #0ea5e9, #6366f1);
    color: white;
    border: none;
    border-radius: 8px;
    font-family: 'Syne', sans-serif;
    font-weight: 600;
    font-size: 0.9rem;
    padding: 0.6rem 1.5rem;
    letter-spacing: 1px;
    transition: opacity 0.2s;
    width: 100%;
}

.stButton > button:hover { opacity: 0.85; }

/* Sliders & Inputs */
.stSlider [data-baseweb="slider"] { accent-color: #38bdf8; }

div[data-testid="stNumberInput"] input {
    background: #111827;
    border: 1px solid #1e3a5f;
    color: #e2e8f0;
    border-radius: 6px;
    font-family: 'DM Mono', monospace;
}

/* DataFrame */
.stDataFrame { border: 1px solid #1e3a5f; border-radius: 8px; }

/* Tabs */
button[data-baseweb="tab"] {
    font-family: 'Syne', sans-serif;
    font-weight: 600;
    font-size: 0.85rem;
    letter-spacing: 1px;
    color: #64748b;
}

button[data-baseweb="tab"][aria-selected="true"] { color: #38bdf8 !important; }

/* Divider */
hr { border-color: #1e2d4a; }

/* Selectbox */
.stSelectbox div[data-baseweb="select"] {
    background: #111827;
    border: 1px solid #1e3a5f;
}
</style>
""", unsafe_allow_html=True)


# ─── Data Generation (Simulated realistic fraud data) ────────────────────────
@st.cache_data
def generate_data(n_samples=10000):
    np.random.seed(42)
    n_fraud = int(n_samples * 0.02)
    n_legit = n_samples - n_fraud

    # Legitimate transactions
    legit = pd.DataFrame({
        'Amount': np.random.exponential(80, n_legit),
        'Hour': np.random.randint(6, 23, n_legit),
        'V1': np.random.normal(0, 1, n_legit),
        'V2': np.random.normal(0, 1, n_legit),
        'V3': np.random.normal(0, 1, n_legit),
        'V4': np.random.normal(0, 1, n_legit),
        'V5': np.random.normal(0, 1, n_legit),
        'Class': 0
    })

    # Fraudulent transactions (distinct pattern)
    fraud = pd.DataFrame({
        'Amount': np.random.exponential(300, n_fraud),
        'Hour': np.random.choice([1, 2, 3, 4, 23], n_fraud),
        'V1': np.random.normal(-3, 1.5, n_fraud),
        'V2': np.random.normal(2, 1.5, n_fraud),
        'V3': np.random.normal(-2, 1.5, n_fraud),
        'V4': np.random.normal(3, 1.5, n_fraud),
        'V5': np.random.normal(-1, 2, n_fraud),
        'Class': 1
    })

    df = pd.concat([legit, fraud]).sample(frac=1, random_state=42).reset_index(drop=True)
    return df


@st.cache_resource
def train_models(df):
    X = df.drop('Class', axis=1)
    y = df['Class']

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42, stratify=y
    )

    # SMOTE
    sm = SMOTE(random_state=42)
    X_res, y_res = sm.fit_resample(X_train, y_train)

    # Models
    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    }

    results = {}
    for name, model in models.items():
        model.fit(X_res, y_res)
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]
        report = classification_report(y_test, y_pred, output_dict=True)
        results[name] = {
            "model": model,
            "y_test": y_test,
            "y_pred": y_pred,
            "y_prob": y_prob,
            "report": report,
            "roc_auc": roc_auc_score(y_test, y_prob),
            "cm": confusion_matrix(y_test, y_pred)
        }

    return results, scaler, X.columns.tolist()


# ─── Plot Helpers ────────────────────────────────────────────────────────────
DARK_BG = "#0a0e1a"
CARD_BG = "#111827"
ACCENT  = "#38bdf8"
PURPLE  = "#818cf8"
PINK    = "#f472b6"
RED     = "#ef4444"
GREEN   = "#22c55e"
TEXT    = "#e2e8f0"
MUTED   = "#64748b"

def style_fig(fig, ax_list=None):
    fig.patch.set_facecolor(DARK_BG)
    axes = ax_list if ax_list else fig.get_axes()
    for ax in axes:
        ax.set_facecolor(CARD_BG)
        ax.tick_params(colors=MUTED, labelsize=8)
        ax.xaxis.label.set_color(MUTED)
        ax.yaxis.label.set_color(MUTED)
        ax.title.set_color(TEXT)
        for spine in ax.spines.values():
            spine.set_edgecolor("#1e2d4a")
    return fig


def plot_class_distribution(df):
    counts = df['Class'].value_counts()
    fig, ax = plt.subplots(figsize=(4, 3))
    bars = ax.bar(['Legitimate', 'Fraud'], counts.values,
                  color=[GREEN, RED], width=0.5, edgecolor='none')
    ax.set_title("Class Distribution", fontsize=11, fontweight='bold', pad=10)
    for bar, val in zip(bars, counts.values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 50,
                f'{val:,}', ha='center', va='bottom', color=TEXT, fontsize=9, fontweight='bold')
    style_fig(fig)
    return fig


def plot_amount_dist(df):
    fig, ax = plt.subplots(figsize=(5, 3))
    df[df.Class==0]['Amount'].clip(upper=1000).plot.hist(
        bins=40, ax=ax, alpha=0.7, color=ACCENT, label='Legitimate')
    df[df.Class==1]['Amount'].clip(upper=1000).plot.hist(
        bins=40, ax=ax, alpha=0.7, color=RED, label='Fraud')
    ax.set_title("Transaction Amount Distribution", fontsize=11, fontweight='bold', pad=10)
    ax.set_xlabel("Amount ($)")
    ax.legend(facecolor=CARD_BG, edgecolor='#1e2d4a', labelcolor=TEXT, fontsize=8)
    style_fig(fig)
    return fig


def plot_hour_dist(df):
    fig, ax = plt.subplots(figsize=(5, 3))
    df[df.Class==0]['Hour'].plot.hist(bins=24, ax=ax, alpha=0.7, color=PURPLE, label='Legitimate')
    df[df.Class==1]['Hour'].plot.hist(bins=24, ax=ax, alpha=0.7, color=PINK, label='Fraud')
    ax.set_title("Transaction Hour Distribution", fontsize=11, fontweight='bold', pad=10)
    ax.set_xlabel("Hour of Day")
    ax.legend(facecolor=CARD_BG, edgecolor='#1e2d4a', labelcolor=TEXT, fontsize=8)
    style_fig(fig)
    return fig


def plot_confusion_matrix(cm, model_name):
    fig, ax = plt.subplots(figsize=(4, 3.5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                ax=ax, cbar=False,
                xticklabels=['Legit', 'Fraud'],
                yticklabels=['Legit', 'Fraud'],
                linewidths=0.5, linecolor='#0a0e1a',
                annot_kws={"size": 14, "weight": "bold", "color": TEXT})
    ax.set_title(f"Confusion Matrix\n{model_name}", fontsize=10, fontweight='bold', pad=10)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    style_fig(fig)
    return fig


def plot_roc(results):
    fig, ax = plt.subplots(figsize=(5, 4))
    colors = [ACCENT, PINK]
    for (name, res), color in zip(results.items(), colors):
        fpr, tpr, _ = roc_curve(res['y_test'], res['y_prob'])
        ax.plot(fpr, tpr, color=color, lw=2,
                label=f"{name} (AUC={res['roc_auc']:.3f})")
    ax.plot([0,1],[0,1], '--', color=MUTED, lw=1)
    ax.set_title("ROC Curve", fontsize=11, fontweight='bold', pad=10)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.legend(facecolor=CARD_BG, edgecolor='#1e2d4a', labelcolor=TEXT, fontsize=8)
    style_fig(fig)
    return fig


def plot_feature_importance(model, feature_names):
    if not hasattr(model, 'feature_importances_'):
        return None
    imp = model.feature_importances_
    idx = np.argsort(imp)[::-1]
    fig, ax = plt.subplots(figsize=(5, 3.5))
    colors_bar = [ACCENT if i < 3 else PURPLE for i in range(len(idx))]
    ax.bar(range(len(idx)), imp[idx], color=colors_bar, edgecolor='none')
    ax.set_xticks(range(len(idx)))
    ax.set_xticklabels([feature_names[i] for i in idx], rotation=45, ha='right', fontsize=7)
    ax.set_title("Feature Importances (Random Forest)", fontsize=11, fontweight='bold', pad=10)
    style_fig(fig)
    return fig


# ─── App Layout ─────────────────────────────────────────────────────────────
def main():
    # Sidebar
    with st.sidebar:
        st.markdown("""
        <div style='padding: 1rem 0;'>
            <div style='font-family:Syne,sans-serif; font-size:1.4rem; font-weight:800;
                        background: linear-gradient(90deg,#38bdf8,#818cf8);
                        -webkit-background-clip:text; -webkit-text-fill-color:transparent;'>
                🛡️ FraudSense
            </div>
            <div style='font-size:0.7rem; color:#64748b; letter-spacing:2px; margin-top:2px;'>
                CREDIT CARD FRAUD DETECTOR
            </div>
        </div>
        <hr style='border-color:#1e2d4a; margin: 0.5rem 0 1.2rem 0;'>
        """, unsafe_allow_html=True)

        st.markdown("<div style='font-size:0.75rem; color:#64748b; letter-spacing:2px; text-transform:uppercase; margin-bottom:0.5rem;'>Dataset Size</div>", unsafe_allow_html=True)
        n_samples = st.slider("", 3000, 20000, 10000, 1000, label_visibility="collapsed")

        st.markdown("<div style='font-size:0.75rem; color:#64748b; letter-spacing:2px; text-transform:uppercase; margin:1rem 0 0.5rem;'>Model</div>", unsafe_allow_html=True)
        chosen_model = st.selectbox("", ["Random Forest", "Logistic Regression"], label_visibility="collapsed")

        st.markdown("<hr style='border-color:#1e2d4a; margin: 1.2rem 0;'>", unsafe_allow_html=True)
        st.markdown("""
        <div style='font-size:0.7rem; color:#475569; line-height:1.6;'>
        <b style='color:#64748b;'>ABOUT</b><br>
        Simulated credit card fraud dataset with SMOTE balancing.
        Binary classification: Fraud vs Legitimate.
        </div>
        """, unsafe_allow_html=True)

    # Header
    st.markdown('<div class="hero-title">Credit Card Fraud Detection</div>', unsafe_allow_html=True)
    st.markdown('<div class="hero-sub">⚡ ML-Powered · SMOTE Balanced · Real-time Prediction</div>', unsafe_allow_html=True)

    # Load & Train
    with st.spinner("🔄 Generating data & training models..."):
        df = generate_data(n_samples)
        results, scaler, feature_names = train_models(df)

    res = results[chosen_model]

    # ── KPI Row ──────────────────────────────────────────────────────────────
    c1, c2, c3, c4 = st.columns(4)
    fraud_count = df['Class'].sum()
    precision = res['report']['1']['precision']
    recall    = res['report']['1']['recall']
    f1        = res['report']['1']['f1-score']

    with c1:
        st.markdown(f"""<div class="metric-card">
            <div class="metric-value">{n_samples:,}</div>
            <div class="metric-label">Total Transactions</div>
        </div>""", unsafe_allow_html=True)
    with c2:
        st.markdown(f"""<div class="metric-card">
            <div class="metric-value" style="color:#ef4444">{fraud_count}</div>
            <div class="metric-label">Fraudulent Cases</div>
        </div>""", unsafe_allow_html=True)
    with c3:
        st.markdown(f"""<div class="metric-card">
            <div class="metric-value" style="color:#818cf8">{res['roc_auc']:.3f}</div>
            <div class="metric-label">ROC-AUC Score</div>
        </div>""", unsafe_allow_html=True)
    with c4:
        st.markdown(f"""<div class="metric-card">
            <div class="metric-value" style="color:#f472b6">{f1:.3f}</div>
            <div class="metric-label">F1 Score (Fraud)</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Tabs ─────────────────────────────────────────────────────────────────
    tab1, tab2, tab3 = st.tabs(["📊  DATA ANALYSIS", "🤖  MODEL RESULTS", "🔍  PREDICT"])

    # ─ Tab 1: EDA ─────────────────────────────────────────────────────────
    with tab1:
        st.markdown('<div class="section-header">Dataset Overview</div>', unsafe_allow_html=True)
        st.dataframe(
            df.describe().round(2).style.background_gradient(cmap='Blues', axis=1),
            use_container_width=True
        )

        st.markdown('<div class="section-header">Visual Exploration</div>', unsafe_allow_html=True)
        col1, col2, col3 = st.columns(3)
        with col1:
            st.pyplot(plot_class_distribution(df), use_container_width=True)
        with col2:
            st.pyplot(plot_amount_dist(df), use_container_width=True)
        with col3:
            st.pyplot(plot_hour_dist(df), use_container_width=True)

        st.markdown('<div class="section-header">Correlation Heatmap</div>', unsafe_allow_html=True)
        fig, ax = plt.subplots(figsize=(8, 4))
        corr = df.corr()
        mask = np.triu(np.ones_like(corr, dtype=bool))
        sns.heatmap(corr, mask=mask, cmap='coolwarm', center=0, ax=ax,
                    linewidths=0.3, linecolor='#0a0e1a', annot=False, cbar_kws={"shrink": 0.7})
        ax.set_title("Feature Correlation Matrix", fontsize=11, fontweight='bold', pad=10)
        style_fig(fig)
        st.pyplot(fig, use_container_width=True)

    # ─ Tab 2: Model ──────────────────────────────────────────────────────
    with tab2:
        st.markdown(f'<div class="section-header">Active Model: {chosen_model}</div>', unsafe_allow_html=True)

        col1, col2 = st.columns([1, 1.3])
        with col1:
            st.markdown("**Classification Report**")
            report_df = pd.DataFrame(res['report']).transpose().round(3)
            st.dataframe(report_df.style.background_gradient(cmap='Blues', subset=['precision','recall','f1-score']),
                         use_container_width=True)
        with col2:
            st.pyplot(plot_confusion_matrix(res['cm'], chosen_model), use_container_width=True)

        st.markdown('<div class="section-header">ROC & Feature Importance</div>', unsafe_allow_html=True)
        col1, col2 = st.columns(2)
        with col1:
            st.pyplot(plot_roc(results), use_container_width=True)
        with col2:
            fig_fi = plot_feature_importance(res['model'], feature_names)
            if fig_fi:
                st.pyplot(fig_fi, use_container_width=True)
            else:
                st.info("Feature importance not available for Logistic Regression.")

        st.markdown('<div class="section-header">Model Comparison</div>', unsafe_allow_html=True)
        comp = []
        for name, r in results.items():
            comp.append({
                "Model": name,
                "ROC-AUC": round(r['roc_auc'], 4),
                "Precision (Fraud)": round(r['report']['1']['precision'], 4),
                "Recall (Fraud)": round(r['report']['1']['recall'], 4),
                "F1 (Fraud)": round(r['report']['1']['f1-score'], 4),
            })
        comp_df = pd.DataFrame(comp).set_index("Model")
        st.dataframe(comp_df.style.highlight_max(color='#1e3a5f', axis=0), use_container_width=True)

    # ─ Tab 3: Predict ─────────────────────────────────────────────────────
    with tab3:
        st.markdown('<div class="section-header">Enter Transaction Details</div>', unsafe_allow_html=True)
        st.markdown("<div style='font-size:0.8rem; color:#64748b; margin-bottom:1rem;'>Adjust the sliders below to simulate a transaction and detect fraud in real-time.</div>", unsafe_allow_html=True)

        col1, col2 = st.columns(2)
        with col1:
            amount = st.slider("💵 Transaction Amount ($)", 0.0, 5000.0, 150.0, 10.0)
            hour   = st.slider("🕐 Hour of Transaction", 0, 23, 14)
            v1     = st.slider("V1 (PCA Feature)", -10.0, 10.0, 0.0, 0.1)
            v2     = st.slider("V2 (PCA Feature)", -10.0, 10.0, 0.0, 0.1)
        with col2:
            v3     = st.slider("V3 (PCA Feature)", -10.0, 10.0, 0.0, 0.1)
            v4     = st.slider("V4 (PCA Feature)", -10.0, 10.0, 0.0, 0.1)
            v5     = st.slider("V5 (PCA Feature)", -10.0, 10.0, 0.0, 0.1)

        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("🔍  ANALYZE TRANSACTION"):
            input_data = np.array([[amount, hour, v1, v2, v3, v4, v5]])
            input_scaled = scaler.transform(input_data)
            prediction = res['model'].predict(input_scaled)[0]
            probability = res['model'].predict_proba(input_scaled)[0][1]

            if prediction == 1:
                st.markdown(f"""
                <div class="fraud-alert">
                    🚨 FRAUD DETECTED &nbsp;|&nbsp; Confidence: {probability*100:.1f}%<br>
                    <span style='font-size:0.85rem; font-weight:400;'>This transaction shows patterns consistent with fraudulent activity.</span>
                </div>""", unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="safe-alert">
                    ✅ TRANSACTION SAFE &nbsp;|&nbsp; Fraud Probability: {probability*100:.1f}%<br>
                    <span style='font-size:0.85rem; font-weight:400;'>This transaction appears to be legitimate.</span>
                </div>""", unsafe_allow_html=True)

            # Probability gauge
            fig, ax = plt.subplots(figsize=(6, 1.2))
            ax.barh(0, probability, color=RED if probability > 0.5 else GREEN, height=0.4)
            ax.barh(0, 1 - probability, left=probability,
                    color='#1e2d4a', height=0.4)
            ax.set_xlim(0, 1)
            ax.set_yticks([])
            ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
            ax.set_xticklabels(['0%', '25%', '50%', '75%', '100%'], color=MUTED, fontsize=8)
            ax.set_title(f"Fraud Probability: {probability*100:.1f}%", color=TEXT, fontsize=10, fontweight='bold')
            ax.axvline(0.5, color=MUTED, linestyle='--', lw=1)
            style_fig(fig)
            st.pyplot(fig, use_container_width=True)

        st.markdown("""
        <div style='margin-top:2rem; padding:1rem; background:#0d1220; border:1px solid #1e2d4a;
                    border-radius:10px; font-size:0.75rem; color:#475569; line-height:1.8;'>
        <b style='color:#64748b;'>💡 FRAUD PATTERNS TO TEST</b><br>
        • High amount ($800+) + Late night hour (1–4 AM) + V1 = -5, V2 = +4 → likely <b style='color:#ef4444;'>FRAUD</b><br>
        • Moderate amount ($50–200) + Daytime (9–18) + V features near 0 → likely <b style='color:#22c55e;'>SAFE</b>
        </div>
        """, unsafe_allow_html=True)

    # Footer
    st.markdown("""
    <hr style='border-color:#1e2d4a; margin-top:3rem;'>
    <div style='text-align:center; font-size:0.7rem; color:#334155; padding:0.5rem 0 1rem;'>
        Built with Python · Scikit-learn · SMOTE · Streamlit &nbsp;|&nbsp; Final Year Data Science Project
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
