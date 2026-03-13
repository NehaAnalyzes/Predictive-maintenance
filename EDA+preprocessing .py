# =============================================================================
# PREDICTIVE MAINTENANCE — PHASE 2 (EDA) + PHASE 3 (PREPROCESSING)
# Dataset : AI4I 2020 Predictive Maintenance (ai4i2020.csv)
# Run     : python phase2_and_3.py
# Installs: pip install pandas numpy matplotlib seaborn scikit-learn imbalanced-learn joblib
# =============================================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
import joblib
import warnings
warnings.filterwarnings('ignore')

plt.rcParams.update({'figure.facecolor': '#f9f9f9', 'axes.grid': True, 'font.family': 'monospace'})

# =============================================================================
# PHASE 2 — EDA
# =============================================================================
print("\n" + "="*60)
print("PHASE 2 — EXPLORATORY DATA ANALYSIS")
print("="*60)

# --------------------------------------------------------------------------
# 2.1 Load Data
# --------------------------------------------------------------------------
df = pd.read_csv('ai4i2020.csv')
print(f"\n[2.1] Loaded dataset: {df.shape[0]:,} rows × {df.shape[1]} columns")

# --------------------------------------------------------------------------
# 2.2 Data Quality
# --------------------------------------------------------------------------
print("\n[2.2] DATA QUALITY CHECK")
print(f"  Missing values : {df.isnull().sum().sum()}")
print(f"  Duplicate rows : {df.duplicated().sum()}")
print(f"  Dtypes         :\n{df.dtypes.to_string()}")

# --------------------------------------------------------------------------
# 2.3 Class Balance
# --------------------------------------------------------------------------
counts = df['Machine failure'].value_counts()
print(f"\n[2.3] CLASS BALANCE")
print(f"  Normal  (0): {counts[0]:,}  ({counts[0]/len(df)*100:.2f}%)")
print(f"  Failure (1): {counts[1]:,}  ({counts[1]/len(df)*100:.2f}%)")
print(f"  ⚠️  Severe imbalance — use F1/AUC, NOT accuracy")

# --------------------------------------------------------------------------
# 2.4 Sensor Stats: Normal vs Failure
# --------------------------------------------------------------------------
sensors = ['Air temperature [K]', 'Process temperature [K]',
           'Rotational speed [rpm]', 'Torque [Nm]', 'Tool wear [min]']
normal  = df[df['Machine failure'] == 0]
failure = df[df['Machine failure'] == 1]

print(f"\n[2.4] SENSOR MEANS — Normal vs Failure")
print(f"  {'Sensor':<32} {'Normal':>10} {'Failure':>10} {'Diff%':>8}")
print("  " + "-"*62)
for s in sensors:
    mn, mf = normal[s].mean(), failure[s].mean()
    print(f"  {s:<32} {mn:>10.2f} {mf:>10.2f} {(mf-mn)/mn*100:>7.1f}%")

# --------------------------------------------------------------------------
# 2.5 Failure Type Breakdown
# --------------------------------------------------------------------------
print(f"\n[2.5] FAILURE TYPE BREAKDOWN")
for col in ['TWF', 'HDF', 'PWF', 'OSF', 'RNF']:
    print(f"  {col}: {df[col].sum():>4}  ({df[col].sum()/len(df)*100:.2f}%)")

# --------------------------------------------------------------------------
# 2.6 Correlations
# --------------------------------------------------------------------------
print(f"\n[2.6] CORRELATIONS WITH TARGET")
for s in sensors:
    c = df[s].corr(df['Machine failure'])
    print(f"  {s:<32}: {c:+.4f}")

# --------------------------------------------------------------------------
# 2.7 EDA Charts — saved as PNG (no popup needed)
# --------------------------------------------------------------------------
print("\n[2.7] Generating EDA charts...")

# --- Chart A: Class Balance ---
fig, axes = plt.subplots(1, 2, figsize=(10, 4))
fig.suptitle('Chart A — Class Balance', fontweight='bold')
axes[0].bar(['Normal', 'Failure'], counts.values, color=['#1a3a6e', '#8b1a1a'], edgecolor='white')
for i, v in enumerate(counts.values):
    axes[0].text(i, v + 50, f'{v:,}\n({v/len(df)*100:.1f}%)', ha='center', fontsize=10)
axes[0].set_title('Class Distribution')

axes[1].pie(counts.values, labels=['Normal', 'Failure'], colors=['#1a3a6e', '#8b1a1a'],
            autopct='%1.1f%%', startangle=90, wedgeprops={'edgecolor': 'white', 'linewidth': 2})
axes[1].set_title('Class Ratio')
plt.tight_layout()
plt.savefig('chartA_class_balance.png', dpi=130, bbox_inches='tight')
plt.close()
print("  Saved: chartA_class_balance.png")

# --- Chart B: Sensor Distributions ---
fig, axes = plt.subplots(1, 5, figsize=(20, 4))
fig.suptitle('Chart B — Sensor Distributions: Normal vs Failure', fontweight='bold', y=1.02)
for i, s in enumerate(sensors):
    axes[i].hist(normal[s],  bins=40, alpha=0.6, color='#1a3a6e', density=True, label='Normal')
    axes[i].hist(failure[s], bins=20, alpha=0.8, color='#8b1a1a', density=True, label='Failure')
    axes[i].set_title(s.split(' [')[0], fontsize=9)
    if i == 0:
        axes[i].legend(fontsize=8)
plt.tight_layout()
plt.savefig('chartB_sensor_distributions.png', dpi=130, bbox_inches='tight')
plt.close()
print("  Saved: chartB_sensor_distributions.png")

# --- Chart C: Correlation Heatmap ---
feat_cols = sensors + ['Machine failure']
corr = df[feat_cols].corr()
short = ['AirTemp', 'ProcTemp', 'RPM', 'Torque', 'ToolWear', 'FAILURE']
plt.figure(figsize=(8, 6))
sns.heatmap(corr, annot=True, fmt='.3f', cmap='RdBu_r', center=0,
            xticklabels=short, yticklabels=short, linewidths=0.5)
plt.title('Chart C — Correlation Matrix', fontweight='bold')
plt.tight_layout()
plt.savefig('chartC_correlation.png', dpi=130, bbox_inches='tight')
plt.close()
print("  Saved: chartC_correlation.png")

# --- Chart D: Scatter Plots ---
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
fig.suptitle('Chart D — Failure Clusters in Sensor Space', fontweight='bold', y=1.02)
pairs = [
    ('Torque [Nm]',     'Rotational speed [rpm]', 'Torque vs RPM'),
    ('Tool wear [min]', 'Torque [Nm]',             'ToolWear vs Torque'),
    ('Process temperature [K]', 'Air temperature [K]', 'ProcTemp vs AirTemp'),
]
for i, (x, y_col, title) in enumerate(pairs):
    axes[i].scatter(normal[x],  normal[y_col],  c='#1a3a6e', alpha=0.07, s=4,  rasterized=True)
    axes[i].scatter(failure[x], failure[y_col], c='#8b1a1a', alpha=0.9,  s=20,
                    edgecolors='white', linewidths=0.4, zorder=5, label='Failure')
    axes[i].set_xlabel(x.split(' [')[0], fontsize=9)
    axes[i].set_ylabel(y_col.split(' [')[0], fontsize=9)
    axes[i].set_title(title)
    axes[i].legend(fontsize=8)
plt.tight_layout()
plt.savefig('chartD_scatter.png', dpi=130, bbox_inches='tight')
plt.close()
print("  Saved: chartD_scatter.png")

print("\n✅ PHASE 2 COMPLETE")

# =============================================================================
# PHASE 3 — PREPROCESSING
# =============================================================================
print("\n" + "="*60)
print("PHASE 3 — FEATURE ENGINEERING & PREPROCESSING")
print("="*60)

# fresh load — always preprocess from raw
df = pd.read_csv('ai4i2020.csv')

# --------------------------------------------------------------------------
# 3.1 Drop Useless Columns
# --------------------------------------------------------------------------
cols_to_drop = ['UDI', 'Product ID', 'TWF', 'HDF', 'PWF', 'OSF', 'RNF']
df.drop(columns=cols_to_drop, inplace=True)
print(f"\n[3.1] Dropped: {cols_to_drop}")
print(f"      Shape now: {df.shape}")

# --------------------------------------------------------------------------
# 3.2 Encode Type Column
# --------------------------------------------------------------------------
type_map = {'L': 0, 'M': 1, 'H': 2}
df['Type'] = df['Type'].map(type_map)
print(f"\n[3.2] Type encoded → L=0, M=1, H=2")
print(f"      Value counts:\n{df['Type'].value_counts().sort_index().to_string()}")

# --------------------------------------------------------------------------
# 3.3 Feature Engineering
# --------------------------------------------------------------------------
df['power']     = df['Torque [Nm]'] * (df['Rotational speed [rpm]'] * 2 * np.pi / 60)
df['temp_diff'] = df['Process temperature [K]'] - df['Air temperature [K]']
df['strain']    = df['Tool wear [min]'] * df['Torque [Nm]']

print(f"\n[3.3] Engineered features created:")
print(f"      power     = torque × (rpm × 2π/60)   | mean = {df['power'].mean():.0f} W")
print(f"      temp_diff = process_temp - air_temp   | mean = {df['temp_diff'].mean():.2f} K")
print(f"      strain    = tool_wear × torque        | mean = {df['strain'].mean():.0f}")

print(f"\n      Correlations with target:")
for f in ['power', 'temp_diff', 'strain']:
    c = df[f].corr(df['Machine failure'])
    print(f"      {f:<12}: {c:+.4f}")

# --------------------------------------------------------------------------
# 3.4 Define X and y
# --------------------------------------------------------------------------
X = df.drop(columns=['Machine failure'])
y = df['Machine failure']
print(f"\n[3.4] X shape: {X.shape}  |  y shape: {y.shape}")
print(f"      Features: {list(X.columns)}")

# --------------------------------------------------------------------------
# 3.5 Train / Test Split
# --------------------------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"\n[3.5] Train/Test Split (80/20, stratified)")
print(f"      Train: {X_train.shape[0]:,} rows  |  failure rate: {y_train.mean()*100:.2f}%")
print(f"      Test : {X_test.shape[0]:,}  rows  |  failure rate: {y_test.mean()*100:.2f}%")

# --------------------------------------------------------------------------
# 3.6 Scale Features
# --------------------------------------------------------------------------
scaler = StandardScaler()
X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns)
X_test_scaled  = pd.DataFrame(scaler.transform(X_test),      columns=X_test.columns)

print(f"\n[3.6] StandardScaler applied")
print(f"      fit() on train only — transform() on both")
print(f"      X_train mean after scaling: {X_train_scaled.mean().mean():.6f} (≈ 0)")
print(f"      X_train std  after scaling: {X_train_scaled.std().mean():.6f}  (≈ 1)")

# --------------------------------------------------------------------------
# 3.7 SMOTE — Balance Training Set
# --------------------------------------------------------------------------
print(f"\n[3.7] SMOTE — Balancing Training Set")
print(f"      Before → Normal: {(y_train==0).sum():,}  Failure: {(y_train==1).sum():,}")

smote = SMOTE(random_state=42)
X_train_sm, y_train_sm = smote.fit_resample(X_train_scaled, y_train)
y_train_sm = pd.Series(y_train_sm)

print(f"      After  → Normal: {(y_train_sm==0).sum():,}  Failure: {(y_train_sm==1).sum():,}")
print(f"      ✅ SMOTE applied to training set ONLY")

# --------------------------------------------------------------------------
# 3.8 SMOTE Chart
# --------------------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(10, 4))
fig.suptitle('Chart E — SMOTE Effect on Training Set', fontweight='bold')
before = y_train.value_counts()
after  = y_train_sm.value_counts()
axes[0].bar(['Normal', 'Failure'], before.values, color=['#1a3a6e', '#8b1a1a'], edgecolor='white')
for i, v in enumerate(before.values):
    axes[0].text(i, v + 30, f'{v:,}', ha='center', fontweight='bold')
axes[0].set_title('Before SMOTE')
axes[1].bar(['Normal', 'Failure'], after.values,  color=['#1a3a6e', '#1a5c2a'], edgecolor='white')
for i, v in enumerate(after.values):
    axes[1].text(i, v + 30, f'{v:,}', ha='center', fontweight='bold')
axes[1].set_title('After SMOTE')
plt.tight_layout()
plt.savefig('chartE_smote.png', dpi=130, bbox_inches='tight')
plt.close()
print("\n      Saved: chartE_smote.png")

# --------------------------------------------------------------------------
# 3.9 Save Everything
# --------------------------------------------------------------------------
X_train_sm_df = pd.DataFrame(X_train_sm, columns=X_train.columns)
X_train_sm_df.to_csv('X_train.csv', index=False)
X_test_scaled.to_csv('X_test.csv',  index=False)
y_train_sm.to_csv('y_train.csv',    index=False)
y_test.to_csv('y_test.csv',         index=False)
joblib.dump(scaler, 'scaler.pkl')

print(f"\n[3.9] Saved output files:")
print(f"      X_train.csv  → {X_train_sm_df.shape}")
print(f"      X_test.csv   → {X_test_scaled.shape}")
print(f"      y_train.csv  → {y_train_sm.shape}")
print(f"      y_test.csv   → {y_test.shape}")
print(f"      scaler.pkl   → StandardScaler object")

# --------------------------------------------------------------------------
# 3.10 Final Summary
# --------------------------------------------------------------------------
print("\n" + "="*60)
print("FINAL SUMMARY")
print("="*60)
print(f"  Features          : {list(X_train.columns)}")
print(f"  X_train (balanced): {X_train_sm_df.shape}")
print(f"  X_test  (real)    : {X_test_scaled.shape}")
print(f"  y_train balanced  : Normal={( y_train_sm==0).sum():,}  Failure={(y_train_sm==1).sum():,}")
print(f"  y_test  real dist : Normal={(y_test==0).sum():,}   Failure={(y_test==1).sum():,}")
print(f"\n  Charts saved      : chartA → chartE (5 PNG files)")
print(f"  Data saved        : X_train, X_test, y_train, y_test, scaler.pkl")
print(f"\n✅ PHASE 2 + 3 COMPLETE — Ready for Phase 4 (Modeling)")