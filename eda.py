# ============================================================
#   Credit Card Fraud Detection — EDA (Exploratory Data Analysis)
#   Run this to understand the dataset before training
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")

plt.style.use("dark_background")
BLUE = "#3b82f6"
RED  = "#ef4444"
GREEN= "#22c55e"

# ─────────────────────────────────────────────
# LOAD DATA
# ─────────────────────────────────────────────
print("Loading data...")
df = pd.read_csv("creditcard.csv")

print("\n📋 Basic Info:")
print(df.info())
print("\n📊 First 5 rows:")
print(df.head())
print("\n📈 Class distribution:")
print(df["Class"].value_counts())
print(f"\nFraud percentage: {df['Class'].mean()*100:.4f}%")

fig, axes = plt.subplots(2, 2, figsize=(14, 10), facecolor="#0a0e1a")
fig.suptitle("Credit Card Fraud — EDA", fontsize=16, color="white", fontweight="bold")

# ── Plot 1: Class Distribution ──
ax = axes[0, 0]
counts = df["Class"].value_counts()
bars = ax.bar(["Normal (0)", "Fraud (1)"], counts.values, color=[GREEN, RED], width=0.5, edgecolor="none")
ax.set_title("Class Distribution", color="white")
ax.set_facecolor("#0a0e1a")
for bar, val in zip(bars, counts.values):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 500,
            f"{val:,}", ha="center", color="white", fontsize=11)
ax.tick_params(colors="white")
ax.spines[:].set_color("#1e293b")

# ── Plot 2: Transaction Amount Distribution ──
ax = axes[0, 1]
ax.hist(df[df["Class"]==0]["Amount"], bins=80, color=GREEN, alpha=0.6, label="Normal", density=True)
ax.hist(df[df["Class"]==1]["Amount"], bins=80, color=RED,   alpha=0.8, label="Fraud",  density=True)
ax.set_title("Transaction Amount Distribution", color="white")
ax.set_xlabel("Amount ($)", color="#94a3b8")
ax.set_facecolor("#0a0e1a")
ax.legend(facecolor="#1a1f35", labelcolor="white")
ax.tick_params(colors="white")
ax.spines[:].set_color("#1e293b")
ax.set_xlim(0, 1000)

# ── Plot 3: Transactions Over Time ──
ax = axes[1, 0]
df["Hour"] = (df["Time"] // 3600) % 24
hourly = df.groupby(["Hour", "Class"]).size().unstack(fill_value=0)
ax.plot(hourly.index, hourly[0], color=GREEN, label="Normal",  linewidth=2)
ax.plot(hourly.index, hourly[1]*10, color=RED, label="Fraud ×10", linewidth=2, linestyle="--")
ax.set_title("Transactions by Hour of Day", color="white")
ax.set_xlabel("Hour", color="#94a3b8")
ax.set_facecolor("#0a0e1a")
ax.legend(facecolor="#1a1f35", labelcolor="white")
ax.tick_params(colors="white")
ax.spines[:].set_color("#1e293b")

# ── Plot 4: V14 Distribution (most important feature) ──
ax = axes[1, 1]
ax.hist(df[df["Class"]==0]["V14"], bins=80, color=GREEN, alpha=0.6, label="Normal", density=True)
ax.hist(df[df["Class"]==1]["V14"], bins=80, color=RED,   alpha=0.8, label="Fraud",  density=True)
ax.set_title("V14 Feature Distribution\n(Top Fraud Indicator)", color="white")
ax.set_xlabel("V14 Value", color="#94a3b8")
ax.set_facecolor("#0a0e1a")
ax.legend(facecolor="#1a1f35", labelcolor="white")
ax.tick_params(colors="white")
ax.spines[:].set_color("#1e293b")

plt.tight_layout()
plt.savefig("eda_plots.png", dpi=150, bbox_inches="tight", facecolor="#0a0e1a")
print("\n✅ EDA plots saved to eda_plots.png")
plt.show()
