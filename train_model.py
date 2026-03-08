# ============================================================
#   Credit Card Fraud Detection - Model Training Script
#   Run this ONCE to train and save the model
# ============================================================

import pandas as pd
import numpy as np
import pickle
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report, confusion_matrix,
    roc_auc_score, precision_score, recall_score, f1_score
)
from imblearn.over_sampling import SMOTE
import warnings
warnings.filterwarnings("ignore")


# ─────────────────────────────────────────────
# 1. LOAD DATA
# ─────────────────────────────────────────────
print("📂 Loading dataset...")
# Download from: https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud
# Place creditcard.csv in the same folder as this script
df = pd.read_csv("creditcard.csv")
print(f"   Shape: {df.shape}")
print(f"   Fraud cases : {df['Class'].sum()} ({df['Class'].mean()*100:.4f}%)")
print(f"   Normal cases: {(df['Class'] == 0).sum()}")


# ─────────────────────────────────────────────
# 2. FEATURE ENGINEERING
# ─────────────────────────────────────────────
print("\n⚙️  Preparing features...")
df["Hour"] = (df["Time"] // 3600) % 24          # extract hour of day

# Scale Amount and Hour (V1-V28 are already PCA-scaled)
scaler = StandardScaler()
df["Amount_Scaled"] = scaler.fit_transform(df[["Amount"]])
df["Hour_Scaled"]   = scaler.fit_transform(df[["Hour"]])

feature_cols = [f"V{i}" for i in range(1, 29)] + ["Amount_Scaled", "Hour_Scaled"]
X = df[feature_cols]
y = df["Class"]


# ─────────────────────────────────────────────
# 3. TRAIN / TEST SPLIT
# ─────────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"   Train size: {X_train.shape[0]:,}  |  Test size: {X_test.shape[0]:,}")


# ─────────────────────────────────────────────
# 4. SMOTE – handle class imbalance
# ─────────────────────────────────────────────
print("\n🔁 Applying SMOTE to balance classes...")
smote = SMOTE(random_state=42)
X_train_res, y_train_res = smote.fit_resample(X_train, y_train)
print(f"   After SMOTE → Fraud: {y_train_res.sum():,}  Normal: {(y_train_res==0).sum():,}")


# ─────────────────────────────────────────────
# 5. TRAIN MODELS
# ─────────────────────────────────────────────
print("\n🤖 Training Logistic Regression...")
lr = LogisticRegression(max_iter=1000, random_state=42)
lr.fit(X_train_res, y_train_res)

print("🌲 Training Random Forest...")
rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
rf.fit(X_train_res, y_train_res)


# ─────────────────────────────────────────────
# 6. EVALUATE
# ─────────────────────────────────────────────
def evaluate(name, model, X_test, y_test):
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    print(f"\n── {name} ──")
    print(f"   Precision : {precision_score(y_test, y_pred):.4f}")
    print(f"   Recall    : {recall_score(y_test, y_pred):.4f}")
    print(f"   F1 Score  : {f1_score(y_test, y_pred):.4f}")
    print(f"   ROC-AUC   : {roc_auc_score(y_test, y_prob):.4f}")
    print(classification_report(y_test, y_pred, target_names=["Normal","Fraud"]))

print("\n📊 Evaluation Results:")
evaluate("Logistic Regression", lr, X_test, y_test)
evaluate("Random Forest",       rf, X_test, y_test)


# ─────────────────────────────────────────────
# 7. SAVE ARTIFACTS
# ─────────────────────────────────────────────
print("\n💾 Saving model artifacts...")
os.makedirs("model_artifacts", exist_ok=True)

with open("model_artifacts/random_forest.pkl",  "wb") as f: pickle.dump(rf,     f)
with open("model_artifacts/logistic_reg.pkl",   "wb") as f: pickle.dump(lr,     f)
with open("model_artifacts/scaler.pkl",         "wb") as f: pickle.dump(scaler, f)
with open("model_artifacts/feature_cols.pkl",   "wb") as f: pickle.dump(feature_cols, f)

print("   ✅ Saved to model_artifacts/")
print("\n🎉 Training complete! Now run:  streamlit run app.py")
