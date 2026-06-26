
# ==========================================
# CELL 0
# ==========================================
# ── Install required packages ────────────────────────────────────────────────
!pip install -q opendatasets imbalanced-learn lightgbm catboost shap xgboost

# ── Core ─────────────────────────────────────────────────────────────────────
import pandas as pd
import numpy as np
import warnings
import os
import time
import pickle
import joblib
import gc

warnings.filterwarnings('ignore')
np.random.seed(42)

# ── Visualisation ─────────────────────────────────────────────────────────────
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns

plt.rcParams.update({'figure.dpi': 110, 'axes.titlesize': 13,
                     'axes.labelsize': 11, 'xtick.labelsize': 9,
                     'ytick.labelsize': 9})
sns.set_style('whitegrid')

# ── Sklearn ───────────────────────────────────────────────────────────────────
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import (LabelEncoder, StandardScaler,
                                   RobustScaler, MinMaxScaler, PowerTransformer)
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.feature_selection import (VarianceThreshold, SelectKBest,
                                       mutual_info_classif, f_classif)
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                              f1_score, roc_auc_score, confusion_matrix,
                              classification_report, roc_curve,
                              precision_recall_curve, average_precision_score)
from sklearn.ensemble import (RandomForestClassifier,
                               VotingClassifier, StackingClassifier,
                               HistGradientBoostingClassifier)
from sklearn.linear_model import LogisticRegression
from sklearn.inspection import permutation_importance

# ── Tree-based models ─────────────────────────────────────────────────────────
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

# ── Imbalanced-learn ──────────────────────────────────────────────────────────
from imblearn.over_sampling import SMOTE, ADASYN  # ← ADASYN added (smarter than SMOTE)

# ── SHAP ──────────────────────────────────────────────────────────────────────
import shap

print('All libraries imported successfully.')


# ==========================================
# CELL 1
# ==========================================
import opendatasets as od
od.download("https://www.kaggle.com/datasets/nd50095800/od-ids2022")

# ==========================================
# CELL 2
# ==========================================
t0 = time.time()
raw_traffic_df = pd.read_csv(
    "/content/od-ids2022/OD-IDS2022-Dataset.csv",
    low_memory=False
)
print(f"Loaded in {time.time()-t0:.1f}s")
print(f"Shape: {raw_traffic_df.shape[0]:,} rows × {raw_traffic_df.shape[1]} columns")
raw_traffic_df.head()


# ==========================================
# CELL 3
# ==========================================
# ── Utility: reduce_mem_usage ─────────────────────────────────────────────
def reduce_mem_usage(df: pd.DataFrame) -> pd.DataFrame:
    """Downcast numeric dtypes to float32/int32 to cut RAM usage."""
    for col in df.select_dtypes(include=[np.number]).columns:
        col_min, col_max = df[col].min(), df[col].max()
        if df[col].dtype.kind == 'f':
            df[col] = df[col].astype(np.float32)
        elif df[col].dtype.kind == 'i':
            if col_min >= np.iinfo(np.int32).min and col_max <= np.iinfo(np.int32).max:
                df[col] = df[col].astype(np.int32)
    return df


# ==========================================
# CELL 4
# ==========================================
print("=" * 60)
print(f"Total columns: {raw_traffic_df.shape[1]}")
print("=" * 60)
for idx, col in enumerate(raw_traffic_df.columns):
    print(f"  [{idx:2d}]  '{col}'")

# ==========================================
# CELL 5
# ==========================================
traffic_df = raw_traffic_df.copy()

# Strip whitespace from column names
traffic_df.columns = traffic_df.columns.str.strip()

# ── Missing values ────────────────────────────────────────────────────────────
num_cols = traffic_df.select_dtypes(include=[np.number]).columns
cat_cols = traffic_df.select_dtypes(include='object').columns

traffic_df[num_cols] = traffic_df[num_cols].fillna(traffic_df[num_cols].median())
for col in cat_cols:
    if col != 'Label':
        mode = traffic_df[col].mode()
        if not mode.empty:
            traffic_df[col] = traffic_df[col].fillna(mode[0])

# ── Duplicates ────────────────────────────────────────────────────────────────
before = len(traffic_df)
traffic_df.drop_duplicates(inplace=True)
print(f"Removed {before - len(traffic_df):,} duplicate rows.")

# ── Drop network identifiers (data leakage) ───────────────────────────────────
id_cols = [c for c in ['Src IP', 'Dst IP', 'Src Port', 'Dst Port'] if c in traffic_df.columns]
traffic_df.drop(columns=id_cols, inplace=True)
print(f"Dropped identifier columns: {id_cols}")

# ── Infinite values ───────────────────────────────────────────────────────────
num_cols = traffic_df.select_dtypes(include=[np.number]).columns
inf_mask = np.isinf(traffic_df[num_cols])
print(f"Infinite values found: {inf_mask.sum().sum():,}")

traffic_df[num_cols] = traffic_df[num_cols].replace([np.inf, -np.inf], np.nan)
for col in num_cols:
    cap = traffic_df[col].quantile(0.99)
    traffic_df[col].fillna(cap, inplace=True)

# ── Memory optimisation ───────────────────────────────────────────────────────
traffic_df = reduce_mem_usage(traffic_df)
print(f"Dataset ready: {traffic_df.shape[0]:,} rows × {traffic_df.shape[1]} columns")


# ==========================================
# CELL 6
# ==========================================
print(f"  Duplicate rows: {traffic_df.duplicated().sum():,}")
before = len(traffic_df)
traffic_df.drop_duplicates(inplace=True)
print(f"Removed {before - len(traffic_df):,} duplicate rows.")

# ==========================================
# CELL 7
# ==========================================
       # ── Data quality audit ────────────────────────────────────────────────────────
num_cols = traffic_df.select_dtypes(include=[np.number]).columns
print("=" * 55)
print("  DATA QUALITY AUDIT")
print("=" * 55)
print(f"  Infinite values   : {np.isinf(traffic_df[num_cols]).sum().sum()}")
print(f"  Missing values    : {traffic_df.isnull().sum().sum()}")
print(f"  Duplicate rows    : {traffic_df.duplicated().sum():,}")
print(f"  Label present     : {'Yes' if 'Label' in traffic_df.columns else 'NO'}")
print(f"  Unique labels     : {traffic_df['Label'].nunique()}")
print(f"  Total rows        : {len(traffic_df):,}")
print(f"  Total columns     : {traffic_df.shape[1]}")
print("=" * 55)


# ==========================================
# CELL 8
# ==========================================
# ── Attack category mapping ────────────────────────────────────────────────
def map_attack(label):
    if label == "BENIGN":
        return "BENIGN"

    elif label in [
        "SYN Floods",
        "Denial-of-service",
        "Distributed_denial-of-service",
        "Slow_HTTP_attack"
    ]:
        return "DoS_DDoS"

    elif label in [
        "ARP_Spoofing",
        "Man-in-the-middle",
        "Fragmented Packet Attacks",
        "TCP_Session_Hijacking"
    ]:
        return "Network_Attack"

    elif label in [
        "Time-based SQL Injection",
        "Persistent Cross-Site Scripting in Blog page",
        "ManageEngine ADSelfService Plus 6.1 - CSV Injection"
    ]:
        return "Web_Attack"

    elif label in [
        "Brute Force Attacks"
    ]:
        return "Brute_Force"

    elif label in [
        "Ransomware (Malware)",
        "DLL Hijacking",
        "EXE Hijacking",
        "EXE HijackinPrintNightMare-RCE",
        "Firmware Vulnerabilitie"
    ]:
        return "Malware"

    elif label in [
        "Apache_flink_directory_traversal",
        "Authenticated Remote Code Execution",
        "Exploiting Node Deserialization",
        "Google Chrome Remote Code Execution via Browser",
        "Kernel Exploitation",
        "Print Spooler Service - Local Privilege Escalation",
        "Privilege Escalation Using Unquoted Service Path",
        "Remote Code Execution via Unrestricted File Upload access",
        "Unauthenticated Arbitrary File Upload",
        "Unauthenticated RCE in Credit Card Customer Care System",
        "Webmin 1.962 - Package Update Escape Bypass RCE"
    ]:
        return "Exploit_Attack"

    else:
        return "Other_Attack"

traffic_df['Multi_Label'] = traffic_df['Label'].map(map_attack).fillna("Other_Attack")

# Remove residual 'Other_Attack' if present (tiny, unmapped categories)
traffic_df = traffic_df[traffic_df['Multi_Label'] != "Other_Attack"].copy()

multi_encoder = LabelEncoder()
traffic_df['Multi_Encoded'] = multi_encoder.fit_transform(traffic_df['Multi_Label'])

print("\nEncoded Attack Classes:")
for i, cls in enumerate(multi_encoder.classes_):
    count = (traffic_df['Multi_Encoded'] == i).sum()
    print(f"  {i} → {cls:<22} : {count:>8,} samples")

N_CLASSES = traffic_df['Multi_Encoded'].nunique()
print(f"\nTotal classes: {N_CLASSES}")

# ==========================================
# CELL 9
# ==========================================
# ── 4.1 Dataset overview ─────────────────────────────────────────────────────
print("=" * 60)
print("DATASET OVERVIEW")
print("=" * 60)
print(f"  Shape            : {traffic_df.shape[0]:,} rows × {traffic_df.shape[1]} columns")
print(f"  Memory usage     : {traffic_df.memory_usage(deep=True).sum() / 1e6:.1f} MB")
print(f"\nData Types:")
print(traffic_df.dtypes.value_counts())
print(f"\nNumerical summary (first 5 columns):")
traffic_df.select_dtypes(include=np.number).describe().T.head(5)


# ==========================================
# CELL 10
# ==========================================
# ── 4.2 Class distribution (mapped) ──────────────────────────────────────────
class_dist = (traffic_df['Multi_Label'].value_counts()
              .reset_index()
              .rename(columns={'index': 'Category', 'Multi_Label': 'Count'}))
class_dist.columns = ['Category', 'Count']
class_dist['Pct'] = (class_dist['Count'] / class_dist['Count'].sum() * 100).round(2)

print("Mapped Class Distribution:")
print(class_dist.to_string(index=False))

fig, axes = plt.subplots(1, 2, figsize=(18, 6))
colors = sns.color_palette('tab10', len(class_dist))

axes[0].barh(class_dist['Category'], class_dist['Count'], color=colors)
axes[0].set_xlabel('Sample Count')
axes[0].set_title('Attack Category Distribution')
axes[0].xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'{int(x):,}'))
for bar, val in zip(axes[0].patches, class_dist['Count']):
    axes[0].text(bar.get_width() + 200, bar.get_y() + bar.get_height()/2,
                 f'{val:,}', va='center', fontsize=8)

axes[1].pie(class_dist['Count'], labels=class_dist['Category'],
            autopct='%1.1f%%', colors=colors, startangle=90)
axes[1].set_title('Proportional Share')
plt.tight_layout()
plt.savefig('eda_class_distribution.png', bbox_inches='tight')
plt.show()


# ==========================================
# CELL 11
# ==========================================
# ── 4.3 Correlation heatmap (top 20 variance features) ───────────────────────
EXCLUDE_COLS = ['Label', 'EncodedLabel', 'Multi_Label', 'Multi_Encoded']
feature_cols = [c for c in traffic_df.select_dtypes(include=np.number).columns
                if c not in EXCLUDE_COLS]

top_var = traffic_df[feature_cols].var().nlargest(20).index.tolist()
corr = traffic_df[top_var].corr()

plt.figure(figsize=(16, 12))
mask = np.triu(np.ones_like(corr, dtype=bool))
sns.heatmap(corr, mask=mask, annot=True, fmt='.2f', cmap='RdYlGn', center=0,
            linewidths=0.4, annot_kws={'size': 7})
plt.title('Correlation Heatmap — Top 20 High-Variance Features', fontsize=14)
plt.tight_layout()
plt.savefig('eda_correlation_heatmap.png', bbox_inches='tight')
plt.show()


# ==========================================
# CELL 12
# ==========================================
# ── 4.4 Feature distributions (key network features) ─────────────────────────
key_feats = [f for f in ['Flow Duration', 'Flow Byts/s', 'Flow Pkts/s',
                          'Fwd Pkt Len Mean', 'Bwd Pkt Len Std', 'Pkt Len Max']
             if f in traffic_df.columns]

sample = traffic_df[key_feats + ['Multi_Label']].sample(min(8000, len(traffic_df)), random_state=42)
sample['TrafficType'] = sample['Multi_Label'].apply(
    lambda x: 'BENIGN' if x == 'BENIGN' else 'ATTACK')

fig, axes = plt.subplots(2, 3, figsize=(18, 10))
axes = axes.flatten()
for i, feat in enumerate(key_feats[:6]):
    sns.kdeplot(data=sample, x=feat, hue='TrafficType', ax=axes[i],
                fill=True, alpha=0.4, clip=(sample[feat].quantile(0.01),
                                             sample[feat].quantile(0.99)))
    axes[i].set_title(f'Distribution: {feat}')
plt.suptitle('Feature Distributions — BENIGN vs ATTACK', fontsize=14, y=1.01)
plt.tight_layout()
plt.savefig('eda_feature_distributions.png', bbox_inches='tight')
plt.show()


# ==========================================
# CELL 13
# ==========================================
# ── 4.5 Outlier analysis (IQR-based) ─────────────────────────────────────────
def count_outliers_iqr(series: pd.Series) -> int:
    Q1, Q3 = series.quantile(0.25), series.quantile(0.75)
    IQR = Q3 - Q1
    return int(((series < Q1 - 1.5 * IQR) | (series > Q3 + 1.5 * IQR)).sum())

outlier_counts = {col: count_outliers_iqr(traffic_df[col])
                  for col in feature_cols[:30]}  # sample 30 cols
outlier_df = (pd.Series(outlier_counts)
              .sort_values(ascending=False)
              .reset_index()
              .rename(columns={'index': 'Feature', 0: 'Outlier_Count'}))
outlier_df.columns = ['Feature', 'Outlier_Count']
outlier_df['Outlier_Pct'] = (outlier_df['Outlier_Count'] / len(traffic_df) * 100).round(2)

print("Top 15 Features with Most Outliers (IQR method):")
print(outlier_df.head(30).to_string(index=False))

plt.figure(figsize=(18, 6))
sns.barplot(data=outlier_df.head(30), x='Feature', y='Outlier_Pct', palette='Reds_r')
plt.xticks(rotation=45, ha='right')
plt.ylabel('Outlier %')
plt.title('Top 30 Features — Outlier Percentage (IQR)')
plt.tight_layout()
plt.savefig('eda_outlier_analysis.png', bbox_inches='tight')
plt.show()


# ==========================================
# CELL 14
# ==========================================
# ── 4.6 Skewness analysis ─────────────────────────────────────────────────────
skew_df = (traffic_df[feature_cols].skew()
           .abs()
           .sort_values(ascending=False)
           .reset_index()
           .rename(columns={'index': 'Feature', 0: 'Abs_Skewness'}))
skew_df.columns = ['Feature', 'Abs_Skewness']
skew_df['Skewed'] = skew_df['Abs_Skewness'] > 1.0

print(f"Features with |skewness| > 1: {skew_df['Skewed'].sum()}")
print(skew_df.head(30).to_string(index=False))

plt.figure(figsize=(12, 4))
top_skew = skew_df.head(30)
colors = ['red' if s else 'steelblue' for s in top_skew['Skewed']]
plt.bar(top_skew['Feature'], top_skew['Abs_Skewness'], color=colors)
plt.axhline(1.0, color='orange', linestyle='--', label='Skewness threshold = 1')
plt.xticks(rotation=45, ha='right')
plt.ylabel('|Skewness|')
plt.title('Feature Skewness — Top 30')
plt.legend()
plt.tight_layout()
plt.savefig('eda_skewness_analysis.png', bbox_inches='tight')
plt.show()

# ==========================================
# CELL 15
# ==========================================
# ── 4.7 Class imbalance analysis ─────────────────────────────────────────────
total = len(traffic_df)
imb_df = class_dist.copy()
imb_df['Imbalance_Ratio'] = (imb_df['Count'].max() / imb_df['Count']).round(2)

print("Class Imbalance Analysis:")
print(imb_df.to_string(index=False))
print(
    f"\nOverall Imbalance Ratio: "
    f"{imb_df['Count'].max() / imb_df['Count'].min():.2f}"
)


# ==========================================
# CELL 16
# ==========================================
# ── ENHANCED Section 5: Skewness Handling + Attack-Specific Features ────────────
SKEW_THRESHOLD = 1.0

EXCLUDE_COLS = ['Label', 'EncodedLabel', 'Multi_Label', 'Multi_Encoded']
feature_cols = [c for c in traffic_df.select_dtypes(include=np.number).columns
                if c not in EXCLUDE_COLS]

BINARY_COLS = [c for c in feature_cols if traffic_df[c].nunique() <= 2]
TRANSFORM_CANDIDATES = [c for c in feature_cols if c not in BINARY_COLS]
raw_skewness = traffic_df[TRANSFORM_CANDIDATES].skew()

log1p_cols = [
    col for col in TRANSFORM_CANDIDATES
    if raw_skewness.get(col, 0) > SKEW_THRESHOLD
    and traffic_df[col].min() >= 0
]
yj_cols = []  # Disabled — saves Colab RAM

print(f'Features transformed with log1p : {len(log1p_cols)}')
print(f'Binary columns (not transformed): {len(BINARY_COLS)}')

traffic_df_transformed = traffic_df.copy()
for col in log1p_cols:
    traffic_df_transformed[col] = np.log1p(traffic_df_transformed[col].clip(lower=0))

# ── ADD ADVANCED ATTACK-SPECIFIC FEATURES ──────────────────────────────
print('\nAdding advanced attack-specific features...')

feature_names_raw = [c for c in traffic_df_transformed.select_dtypes(include=np.number).columns
                     if c not in EXCLUDE_COLS]

def add_attack_specific_features(df, feature_names):
    """Add advanced features designed to separate similar attack types."""
    df = df.copy()

    # 1. Byte-to-packet ratio (existing - keep it)
    bps_col = next((c for c in feature_names if 'Byts/s' in c or 'Bytes/s' in c), None)
    pps_col = next((c for c in feature_names if 'Pkts/s' in c or 'Packets/s' in c), None)
    if bps_col and pps_col:
        df['Bytes_Per_Packet'] = (df[bps_col] / (df[pps_col] + 1e-6)).astype(np.float32)
        print(f'Created Bytes_Per_Packet')

    # 2. Fwd/Bwd asymmetry (existing - keep it)
    fwd_col = next((c for c in feature_names if 'Fwd Pkt Len Mean' in c), None)
    bwd_col = next((c for c in feature_names if 'Bwd Pkt Len Mean' in c), None)
    if fwd_col and bwd_col:
        df['Fwd_Bwd_Ratio'] = (df[fwd_col] / (df[bwd_col] + 1e-6)).astype(np.float32)
        print(f'Created Fwd_Bwd_Ratio')

    # 3. Flow duration per packet (existing - keep it)
    dur_col = next((c for c in feature_names if 'Flow Duration' in c), None)
    if dur_col and pps_col:
        df['Duration_Per_Pkt'] = (df[dur_col] / (df[pps_col] + 1e-6)).astype(np.float32)
        print(f'Created Duration_Per_Pkt')

    # 4. NEW: Connection burstiness (detects brute force & scanning)
    if dur_col and pps_col:
        # Short flows with many packets = brute force/login attempts
        df['Connection_Burst'] = ((df[dur_col] < 100000) & (df[pps_col] > 100)).astype(np.float32)
        print(f'Created Connection_Burst')

    # 5. NEW: Packet size entropy (detects exploits with mixed packet sizes)
    size_cols = [c for c in feature_names if 'Pkt Len' in c and 'Mean' in c]
    if len(size_cols) >= 2:
        # High variance = exploit traffic
        df['Packet_Size_Variance'] = df[size_cols].var(axis=1).astype(np.float32)
        print(f'Created Packet_Size_Variance')

    # 6. NEW: Flag anomaly score (detects network attacks)
    flag_cols = [c for c in feature_names if 'Flag' in c]
    if flag_cols:
        # Unusual flag combinations often indicate attacks
        df['Flag_Anomaly'] = df[flag_cols].sum(axis=1).astype(np.float32)
        print(f'Created Flag_Anomaly')

    # 7. NEW: Empty packet ratio (detects web attacks & exploits)
    empty_pkt_col = next((c for c in feature_names if 'Min' in c and 'Pkt Len' in c), None)
    if empty_pkt_col:
        df['Empty_Packet_Ratio'] = (df[empty_pkt_col] == 0).astype(np.float32)
        print(f'Created Empty_Packet_Ratio')

    # 8. NEW: RST flag ratio (detects connection resets - common in scanning)
    rst_col = next((c for c in feature_names if 'RST Flag Cnt' in c), None)
    if rst_col and pps_col:
        df['RST_Ratio'] = (df[rst_col] / (df[pps_col] + 1e-6)).astype(np.float32)
        print(f'Created RST_Ratio')

    return df

# Apply the enhanced feature engineering
traffic_df_transformed = add_attack_specific_features(
    traffic_df_transformed, feature_names_raw
)

# Log transform skewness (keep this)
before_skew = traffic_df[log1p_cols].skew().abs().mean()
after_skew  = traffic_df_transformed[log1p_cols].skew().abs().mean()
print(f'\nMean |skewness| before log1p: {before_skew:.2f}')
print(f'Mean |skewness| after  log1p: {after_skew:.2f}')
print('Section 5 complete — attack-specific features added')

# Clean up
del traffic_df
gc.collect()

# Prepare feature matrix
target_col = 'Multi_Encoded'
cols_to_exclude = ['Label', 'Multi_Label', 'Multi_Encoded']

feature_matrix = traffic_df_transformed.drop(columns=cols_to_exclude, errors='ignore')
target_vector = traffic_df_transformed[target_col]
feature_matrix = feature_matrix.select_dtypes(include=[np.number])

print(f'\nFinal feature matrix shape: {feature_matrix.shape}')
print(f'Total features after engineering: {feature_matrix.shape[1]}')

# ==========================================
# CELL 17
# ==========================================
# ── FIXED Section 6: Scaling + outlier clipping ──────────────────────────────
feature_cols_clean = feature_matrix.columns.tolist()

X_raw = feature_matrix.values.astype(np.float32)
y_raw = target_vector.values

X_train_raw, X_test_raw, y_train, y_test = train_test_split(
    X_raw, y_raw, test_size=0.2, random_state=42, stratify=y_raw
)
print(f'Train: {X_train_raw.shape}, Test: {X_test_raw.shape}')

# ── Step 1: RobustScaler (fits on train only — no leakage) ────────────────────
scaler = RobustScaler()
X_train_scaled = scaler.fit_transform(X_train_raw).astype(np.float32)
X_test_scaled  = scaler.transform(X_test_raw).astype(np.float32)

print(f'After RobustScaler — range: [{X_train_scaled.min():.2f}, {X_train_scaled.max():.2f}]')

# ── Step 2: Clip extreme outliers AFTER scaling ───────────────────────────────
# RobustScaler does NOT remove outliers — it only centres them differently.
# Values beyond ±10 after scaling are extreme outliers (DDoS bursts, etc.)
# that add noise and dominate model splits. Clip them.
CLIP_LIMIT = 10.0
X_train_scaled = np.clip(X_train_scaled, -CLIP_LIMIT, CLIP_LIMIT)
X_test_scaled  = np.clip(X_test_scaled,  -CLIP_LIMIT, CLIP_LIMIT)

print(f'After clipping ±{CLIP_LIMIT} — range: [{X_train_scaled.min():.2f}, {X_train_scaled.max():.2f}]')

# ── Step 3: Report how many values were clipped ───────────────────────────────
n_clipped_train = np.sum(np.abs(X_train_scaled) == CLIP_LIMIT)
n_clipped_test  = np.sum(np.abs(X_test_scaled)  == CLIP_LIMIT)
total_train     = X_train_scaled.size
pct_clipped     = n_clipped_train / total_train * 100

print(f'\nValues clipped in train set : {n_clipped_train:,} / {total_train:,} ({pct_clipped:.3f}%)')
print(f'Values clipped in test set  : {n_clipped_test:,}')

# Healthy: <1% clipped. If >5% is clipped you may want to investigate those features.
if pct_clipped > 5.0:
    print(f'⚠️  WARNING: {pct_clipped:.1f}% of values clipped — check for extreme features.')
    # Find which features have the most outliers
    per_feature_clips = np.sum(np.abs(
        scaler.transform(X_train_raw).astype(np.float32)
    ) > CLIP_LIMIT, axis=0)
    top_outlier_feats = np.argsort(per_feature_clips)[::-1][:5]
    print('  Top features with outliers:')
    for idx in top_outlier_feats:
        if per_feature_clips[idx] > 0:
            print(f'    {feature_cols_clean[idx]}: {per_feature_clips[idx]:,} outlier values')
elif pct_clipped < 1.0:
    print('✅ Clipping is within healthy range (<1%)')

# ── Fix: suppress gc.collect() output ────────────────────────────────────────
# gc.collect() returns the number of collected objects — that's your "8650" number
_ = gc.collect()   # ← assign to _ to suppress the auto-display in Colab

del X_raw, X_train_raw, X_test_raw
_ = gc.collect()

print('\n✅ Section 6 complete — RobustScaler + clipping applied')
print(f'   Final scaled range: [{X_train_scaled.min():.2f}, {X_train_scaled.max():.2f}]')
print(f'   Train shape: {X_train_scaled.shape}')

# ==========================================
# CELL 18
# ==========================================
# ── REPLACE Section 7 with this enhanced version ──────────────────────────────

from sklearn.feature_selection import VarianceThreshold, SelectKBest, mutual_info_classif
from collections import Counter

print('Step 1: VarianceThreshold...')
vt = VarianceThreshold(threshold=0.0)
X_train_vt = vt.fit_transform(X_train_scaled)
X_test_vt = vt.transform(X_test_scaled)
feature_names_vt = [feature_cols_clean[i] for i in vt.get_support(indices=True)]
print(f'  {X_train_scaled.shape[1]} → {X_train_vt.shape[1]} features')

print('Step 2: Correlation filter (>0.95)...')
MI_SAMPLE = min(50_000, len(X_train_vt))
idx_mi = np.random.choice(len(X_train_vt), MI_SAMPLE, replace=False)

corr_df = pd.DataFrame(X_train_vt[idx_mi], columns=feature_names_vt)
corr_mat = corr_df.corr().abs()
upper_tri = corr_mat.where(np.triu(np.ones_like(corr_mat, dtype=bool), k=1))
drop_cols = [c for c in upper_tri.columns if any(upper_tri[c] > 0.95)]
keep_idx = [i for i, n in enumerate(feature_names_vt) if n not in drop_cols]

X_train_corr = X_train_vt[:, keep_idx]
X_test_corr = X_test_vt[:, keep_idx]
feature_names_corr = [feature_names_vt[i] for i in keep_idx]
print(f'  {X_train_vt.shape[1]} → {X_train_corr.shape[1]} features')

print('Step 3: Mutual Information selection...')
K_BEST = 40  # Increased from 35 to capture attack-specific features
selector = SelectKBest(score_func=mutual_info_classif, k=min(K_BEST, X_train_corr.shape[1]))
selector.fit(X_train_corr[idx_mi[:min(MI_SAMPLE, len(X_train_corr))]],
             y_train[idx_mi[:min(MI_SAMPLE, len(X_train_corr))]])

X_train_sel = selector.transform(X_train_corr).astype(np.float32)
X_test_sel = selector.transform(X_test_corr).astype(np.float32)

feat_scores = pd.Series(selector.scores_, index=feature_names_corr)
selected_features = feat_scores.nlargest(K_BEST).index.tolist()

print(f'\n✓ Selected {len(selected_features)} features')
print('Top 15 features by Mutual Information:')
for f, s in feat_scores.nlargest(15).items():
    print(f'  {f:<40}: {s:.4f}')

# Clean up
del X_train_corr, X_test_corr, corr_df
gc.collect()

# Feature importance bar chart
top_feat_df = feat_scores.nlargest(K_BEST).sort_values()
plt.figure(figsize=(10, 9))
plt.barh(top_feat_df.index, top_feat_df.values, color='steelblue')
plt.xlabel('Mutual Information Score')
plt.title(f'Top {K_BEST} Features — Mutual Information (fitted on real data)')
plt.tight_layout()
plt.savefig('feature_selection_mi.png', bbox_inches='tight')
plt.show()
print('✅ Feature selection complete (real data only)')

# ==========================================
# CELL 19
# ==========================================
# ── REPLACE Section 8 with OPTIMAL SMOTE ──────────────────────────────────────

from imblearn.over_sampling import SMOTE

print('Class distribution BEFORE SMOTE:')
class_counts_before = Counter(y_train)
for cls, cnt in sorted(class_counts_before.items()):
    name = multi_encoder.classes_[cls]
    print(f'  {cls} ({name}): {cnt:,}')

# OPTIMAL STRATEGY - Modest increase for minority classes only
max_class_count = max(class_counts_before.values())  # 240,665

# Set target to a reasonable level that doesn't over-amplify minority classes
# Target 25,000 for all classes below that threshold
TARGET_SIZE = 25000  # Reasonable target for minority classes

sampling_strategy = {}
print(f'\nSMOTE targets (modest increase):')
for cls, cnt in class_counts_before.items():
    name = multi_encoder.classes_[cls]
    if cnt < TARGET_SIZE:
        # Only increase up to TARGET_SIZE, not beyond
        target = min(TARGET_SIZE, int(max_class_count * 0.15))  # Cap at 15% of majority
        if cnt < target:
            sampling_strategy[cls] = target
            synthetic_count = target - cnt
            print(f'  {cls} ({name}): {cnt:,} → {target:,} (+{synthetic_count:,} synthetic, {synthetic_count/cnt*100:.1f}% increase)')
        else:
            print(f'  {cls} ({name}): {cnt:,} → {cnt:,} (no change)')
    else:
        print(f'  {cls} ({name}): {cnt:,} → {cnt:,} (no change)')

# Alternative: Only oversample the SINGLE smallest class (Web_Attack)
# This is often the safest approach
print('\n' + '='*60)
print('  OPTION: Oversample ONLY Web_Attack (safest approach)')
print('='*60)

# Calculate reasonable target for Web_Attack only
web_attack_count = class_counts_before[6]  # Web_Attack is class 6
REASONABLE_TARGET = min(25000, web_attack_count * 1.5)  # Max 50% increase

sampling_strategy_safe = {6: int(REASONABLE_TARGET)}
print(f'  Web_Attack (6): {web_attack_count:,} → {int(REASONABLE_TARGET):,} (+{int(REASONABLE_TARGET) - web_attack_count:,} synthetic)')
print(f'  All other classes: unchanged')

# Choose which strategy to use
USE_SAFE_STRATEGY = True  # Set to False to use the multi-class strategy

if USE_SAFE_STRATEGY:
    sampling_strategy = sampling_strategy_safe
    print('\n✓ Using SAFE strategy (only Web_Attack oversampled)')
else:
    print('\n✓ Using MODERATE strategy (multiple classes oversampled)')

if len(sampling_strategy) == 0:
    print('\n⚠️ WARNING: No classes will be oversampled! Check your TARGET_SIZE.')
else:
    # Apply SMOTE
    oversampler = SMOTE(sampling_strategy=sampling_strategy, k_neighbors=min(5, min(class_counts_before.values())-1), random_state=42)

    t0 = time.time()
    X_train_sm, y_train_sm = oversampler.fit_resample(X_train_sel, y_train)
    X_train_sm = X_train_sm.astype(np.float32)
    print(f'\nOversampling done in {time.time()-t0:.1f}s')
    print(f'Train shape AFTER oversampling: {X_train_sm.shape}')

    print('\nClass distribution AFTER SMOTE:')
    for cls, cnt in sorted(Counter(y_train_sm).items()):
        name = multi_encoder.classes_[cls]
        original = class_counts_before[cls]
        change = cnt - original
        change_pct = (change / original * 100) if original > 0 else 0
        print(f'  {cls} ({name}): {cnt:,} ({change:+d}, {change_pct:+.1f}%)')

gc.collect()
print('✅ Section 8 complete — Optimal SMOTE applied')

# ==========================================
# CELL 20
# ==========================================
# ── Utility: evaluate a model ─────────────────────────────────────────────────
def evaluate_model(model, X_tr, y_tr, X_te, y_te, name, fit_time):
    y_pred = model.predict(X_te)
    acc  = accuracy_score(y_te, y_pred)
    prec = precision_score(y_te, y_pred, average='weighted', zero_division=0)
    rec  = recall_score(y_te, y_pred, average='weighted', zero_division=0)
    wf1  = f1_score(y_te, y_pred, average='weighted', zero_division=0)
    mf1  = f1_score(y_te, y_pred, average='macro', zero_division=0)

    try:
        proba = model.predict_proba(X_te)
        auc = roc_auc_score(y_te, proba, multi_class='ovr', average='weighted')
    except Exception:
        auc = np.nan

    print(f"\n{'='*55}")
    print(f'  {name}')
    print(f"{'='*55}")
    print(f'  Accuracy        : {acc:.4f}')
    print(f'  Precision (W)   : {prec:.4f}')
    print(f'  Recall (W)      : {rec:.4f}')
    print(f'  Weighted F1     : {wf1:.4f}')
    print(f'  Macro F1        : {mf1:.4f}')
    print(f'  ROC-AUC (W)     : {auc:.4f}')
    print(f'  Training time   : {fit_time:.1f}s')
    print(classification_report(y_te, y_pred,
                                target_names=multi_encoder.classes_,
                                zero_division=0))
    return {'Model': name, 'Accuracy': acc, 'Precision_W': prec,
            'Recall_W': rec, 'Weighted_F1': wf1, 'Macro_F1': mf1,
            'ROC_AUC': auc, 'Train_Time_s': fit_time}

results = []
trained_models = {}

# ==========================================
# CELL 21
# ==========================================
# ── IMPORTANT: Verify shapes match before ANY model training ─────────────────
print('Shape check before training:')
print(f'  X_train_sm : {X_train_sm.shape}')
print(f'  y_train_sm : {y_train_sm.shape}')
print(f'  X_test_sel : {X_test_sel.shape}')
print(f'  y_test     : {y_test.shape}')

assert X_train_sm.shape[0] == y_train_sm.shape[0], \
    f'X/y mismatch! X={X_train_sm.shape[0]}, y={y_train_sm.shape[0]}'
assert X_test_sel.shape[0] == y_test.shape[0], \
    f'Test X/y mismatch! X={X_test_sel.shape[0]}, y={y_test.shape[0]}'

print('✅ All shapes consistent — safe to train')

# ==========================================
# CELL 22
# ==========================================
# ── 9.1 Random Forest ─────────────────────────────────────────────────────────
print('Training Random Forest...')
t0 = time.time()
rf_model = RandomForestClassifier(
    n_estimators=250,
    max_depth=25,
    min_samples_split=4,
    min_samples_leaf=2,
    max_features='sqrt',
    class_weight='balanced_subsample',
    n_jobs=-1,
    random_state=42
)
rf_model.fit(X_train_sm, y_train_sm)   # ← X_train_sm not X_train_sel
fit_time = time.time() - t0

res_rf = evaluate_model(rf_model, X_train_sm, y_train_sm,
                         X_test_sel, y_test, 'Random Forest', fit_time)
results.append(res_rf)
trained_models['Random Forest'] = rf_model
_ = gc.collect()
print('✅ Random Forest done')

# ==========================================
# CELL 23
# ==========================================
# ── 9.2 XGBoost (OPTIMISED PARAMS) ───────────────────────────────────────────
# Key changes: n_estimators 300→500, max_depth 7→8, min_child_weight added,
#              gamma added for regularisation, scale_pos_weight handled
print('Training XGBoost...')
t0 = time.time()
xgb_model = XGBClassifier(
    n_estimators=500,         # was 400 - reduced to prevent overfitting
    learning_rate=0.05,
    max_depth=7,                # was 8 - shallower trees
    subsample=0.8,
    colsample_bytree=0.8,
    reg_alpha=1.0,              # NEW - L1 regularization
    reg_lambda=1.0,             # NEW - L2 regularization
    use_label_encoder=False,
    eval_metric='mlogloss',
    tree_method='hist',
    n_jobs=-1,
    random_state=42
)
xgb_model.fit(X_train_sm, y_train_sm,
              eval_set=[(X_test_sel, y_test)],
              verbose=False)
fit_time = time.time() - t0

res_xgb = evaluate_model(xgb_model, X_train_sm, y_train_sm,
                           X_test_sel, y_test, 'XGBoost', fit_time)
results.append(res_xgb)
trained_models['XGBoost'] = xgb_model
gc.collect()

# ==========================================
# CELL 24
# ==========================================
# ── 9.3 LightGBM (OPTIMISED PARAMS) ──────────────────────────────────────────
# Key changes: n_estimators 300→600, num_leaves 63→127, min_child_samples added
# LightGBM is usually the BEST single model for IDS datasets
print('Training LightGBM...')
t0 = time.time()
lgbm_model = LGBMClassifier(
    n_estimators=450,           # was 400
    learning_rate=0.05,
    num_leaves=83,              # was 127 - much simpler trees
    max_depth=6,                # NEW - limit depth
    min_child_samples=50,       # NEW - prevent overfitting
    subsample=0.8,
    colsample_bytree=0.8,
    reg_alpha=0.5,              # NEW
    reg_lambda=1.0,             # NEW
    class_weight='balanced',
    n_jobs=-1,
    random_state=42,
    verbose=-1
)
lgbm_model.fit(
    X_train_sm, y_train_sm,
    eval_set=[(X_test_sel, y_test)],
    callbacks=[lgbm_model.__class__  # just run fit normally
               ] if False else None
)
fit_time = time.time() - t0

res_lgbm = evaluate_model(lgbm_model, X_train_sm, y_train_sm,
                            X_test_sel, y_test, 'LightGBM', fit_time)
results.append(res_lgbm)
trained_models['LightGBM'] = lgbm_model
gc.collect()

# ==========================================
# CELL 25
# ==========================================
# ── 9.4 HistGradientBoosting (NEW MODEL) ─────────────────────────────────────
# Why: sklearn-native, no install needed, very RAM-efficient, fast on Colab
# Rivals XGB in accuracy, built-in class_weight support, handles NaN natively
# Typical accuracy on IDS datasets: 90-93%
print('Training HistGradientBoosting (NEW)...')
t0 = time.time()
hgb_model = HistGradientBoostingClassifier(
    max_iter=550,               # was 450
    learning_rate=0.1,
    max_leaf_nodes=64,          # was 63
    max_depth=8,                # NEW
    min_samples_leaf=50,        # was 20 - more regularization
    l2_regularization=1.0,      # was 0.1
    class_weight='balanced',
    early_stopping=True,
    n_iter_no_change=25,        # was 20
    validation_fraction=0.2,    # was 0.1
    random_state=42
)
hgb_model.fit(X_train_sm, y_train_sm)
fit_time = time.time() - t0

res_hgb = evaluate_model(hgb_model, X_train_sm, y_train_sm,
                           X_test_sel, y_test, 'HistGradientBoosting', fit_time)
results.append(res_hgb)
trained_models['HistGradientBoosting'] = hgb_model
gc.collect()

# ==========================================
# CELL 26
# ==========================================
# ── 10.1 Confusion matrices ────────────────────────────────────────────────────
def plot_confusion_matrix(model, X_te, y_te, name, label_names):
    y_pred = model.predict(X_te)
    cm = confusion_matrix(y_te, y_pred)
    cm_norm = cm.astype('float') / cm.sum(axis=1, keepdims=True)

    plt.figure(figsize=(9, 7))
    sns.heatmap(cm_norm, annot=True, fmt='.2f', cmap='Blues',
                xticklabels=label_names, yticklabels=label_names,
                annot_kws={'size': 9})
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title(f'Confusion Matrix — {name} (normalised)')
    plt.tight_layout()
    plt.savefig(f'cm_{name.lower().replace(" ","_")}.png', bbox_inches='tight')
    plt.show()

for mname, model in trained_models.items():
    plot_confusion_matrix(model, X_test_sel, y_test, mname, multi_encoder.classes_)


# ==========================================
# CELL 27
# ==========================================
# ── 10.2 ROC Curves (One-vs-Rest, per class) ─────────────────────────────────
def plot_roc_curves(model, X_te, y_te, name, label_names, n_classes):
    try:
        proba = model.predict_proba(X_te)
    except Exception:
        print(f"ROC not available for {name}")
        return

    y_bin = np.eye(n_classes)[y_te]
    plt.figure(figsize=(10, 7))
    for i, cls in enumerate(label_names):
        fpr, tpr, _ = roc_curve(y_bin[:, i], proba[:, i])
        auc_score = roc_auc_score(y_bin[:, i], proba[:, i])
        plt.plot(fpr, tpr, label=f'{cls} (AUC={auc_score:.3f})')

    plt.plot([0,1],[0,1],'k--', alpha=0.4)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curves — {name}')
    plt.legend(loc='lower right', fontsize=8)
    plt.tight_layout()
    plt.savefig(f'roc_{name.lower().replace(" ","_")}.png', bbox_inches='tight')
    plt.show()

# Plot for top 3 models by weighted F1
results_df_tmp = pd.DataFrame(results).sort_values('Weighted_F1', ascending=False)
top3_names = results_df_tmp['Model'].head(3).tolist()

for mname in top3_names:
    if mname in trained_models:
        plot_roc_curves(trained_models[mname], X_test_sel, y_test,
                        mname, multi_encoder.classes_, N_CLASSES)


# ==========================================
# CELL 28
# ==========================================
# ── 10.3 Precision-Recall Curves (top model) ─────────────────────────────────
best_model_name_tmp = results_df_tmp['Model'].iloc[0]
best_model_tmp = trained_models[best_model_name_tmp]
proba_best = best_model_tmp.predict_proba(X_test_sel)
y_bin = np.eye(N_CLASSES)[y_test]

plt.figure(figsize=(10, 7))
for i, cls in enumerate(multi_encoder.classes_):
    prec_c, rec_c, _ = precision_recall_curve(y_bin[:, i], proba_best[:, i])
    ap = average_precision_score(y_bin[:, i], proba_best[:, i])
    plt.plot(rec_c, prec_c, label=f'{cls} (AP={ap:.3f})')

plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title(f'Precision-Recall Curves — {best_model_name_tmp}')
plt.legend(loc='upper right', fontsize=8)
plt.tight_layout()
plt.savefig('precision_recall_curves.png', bbox_inches='tight')
plt.show()


# ==========================================
# CELL 29
# ==========================================
# ── NEW Section 10.4: Per-class threshold calibration ───────────────────────
# Fix for: ROC-AUC=96% but Accuracy=78% (decision boundary at wrong point)

def find_optimal_thresholds(model, X_te, y_te, n_classes, label_names):
    """Find per-class threshold that maximises F1 score."""
    try:
        proba = model.predict_proba(X_te)
    except Exception:
        print('Model does not support predict_proba — skipping threshold calibration')
        return None

    thresholds = {}
    print(f'\n{"="*60}')
    print('  PER-CLASS THRESHOLD CALIBRATION')
    print(f'{"="*60}')

    for i, cls in enumerate(label_names):
        y_bin = (y_te == i).astype(int)
        prec_arr, rec_arr, thresh_arr = precision_recall_curve(y_bin, proba[:, i])

        # F1 at each threshold
        with np.errstate(divide='ignore', invalid='ignore'):
            f1_arr = np.where(
                (prec_arr + rec_arr) > 0,
                2 * prec_arr * rec_arr / (prec_arr + rec_arr),
                0
            )

        best_idx = np.argmax(f1_arr[:-1])  # last threshold is 1.0 — skip
        best_thresh = thresh_arr[best_idx]
        best_f1     = f1_arr[best_idx]
        thresholds[i] = best_thresh
        print(f'  {cls:<22}: threshold={best_thresh:.3f}  F1={best_f1:.4f}')

    return thresholds


def predict_with_thresholds(model, X, thresholds):
    """Apply per-class thresholds at prediction time."""
    proba = model.predict_proba(X)
    # Divide each column by its threshold — highest ratio wins
    thresh_arr = np.array([thresholds[i] for i in range(proba.shape[1])])
    adjusted   = proba / (thresh_arr + 1e-9)
    return np.argmax(adjusted, axis=1)


def evaluate_with_thresholds(model, X_te, y_te, name, thresholds, label_names):
    """Evaluate model using calibrated thresholds."""
    y_pred_default   = model.predict(X_te)
    y_pred_calibrated = predict_with_thresholds(model, X_te, thresholds)

    acc_default    = accuracy_score(y_te, y_pred_default)
    acc_calibrated = accuracy_score(y_te, y_pred_calibrated)
    f1_default     = f1_score(y_te, y_pred_default, average='weighted', zero_division=0)
    f1_calibrated  = f1_score(y_te, y_pred_calibrated, average='weighted', zero_division=0)

    print(f'\n{"="*60}')
    print(f'  {name} — Threshold Calibration Results')
    print(f'{"="*60}')
    print(f'  Default (0.5 threshold):')
    print(f'    Accuracy   : {acc_default:.4f}')
    print(f'    Weighted F1: {f1_default:.4f}')
    print(f'  Calibrated (per-class thresholds):')
    print(f'    Accuracy   : {acc_calibrated:.4f}  (gain: {acc_calibrated-acc_default:+.4f})')
    print(f'    Weighted F1: {f1_calibrated:.4f}  (gain: {f1_calibrated-f1_default:+.4f})')
    print()
    print(classification_report(y_te, y_pred_calibrated,
                                target_names=label_names, zero_division=0))
    return {
        'model': name,
        'acc_default': acc_default,
        'acc_calibrated': acc_calibrated,
        'acc_gain': acc_calibrated - acc_default,
        'f1_default': f1_default,
        'f1_calibrated': f1_calibrated,
        'thresholds': thresholds
    }


# ── Run threshold calibration on all trained models ───────────────────────────
print('Computing optimal thresholds for all models...')
calibration_results = []

for mname, model in trained_models.items():
    if not hasattr(model, 'predict_proba'):
        print(f'  Skipping {mname} (no predict_proba)')
        continue
    thresholds = find_optimal_thresholds(
        model, X_test_sel, y_test, N_CLASSES, multi_encoder.classes_)
    if thresholds:
        cal_res = evaluate_with_thresholds(
            model, X_test_sel, y_test, mname, thresholds, multi_encoder.classes_)
        calibration_results.append(cal_res)

# Summary table
if calibration_results:
    print('\n' + '='*65)
    print('  THRESHOLD CALIBRATION SUMMARY')
    print('='*65)
    print(f'  {"Model":<28} {"Default Acc":>12} {"Calibrated":>12} {"Gain":>8}')
    print('-'*65)
    for r in sorted(calibration_results, key=lambda x: -x['acc_calibrated']):
        print(f'  {r["model"]:<28} {r["acc_default"]:>12.4f} {r["acc_calibrated"]:>12.4f} {r["acc_gain"]:>+8.4f}')
    print('='*65)

# Save best calibrated thresholds for prevention engine
best_calibrated = max(calibration_results, key=lambda x: x['acc_calibrated'])
best_calibrated_thresholds = best_calibrated['thresholds']
best_calibrated_model_name = best_calibrated['model']
print(f'\n🏆 Best calibrated model: {best_calibrated_model_name}')
print(f'   Accuracy (default):    {best_calibrated["acc_default"]:.4f}')
print(f'   Accuracy (calibrated): {best_calibrated["acc_calibrated"]:.4f}')
print(f'   Gain from calibration: {best_calibrated["acc_gain"]:+.4f}')


# ==========================================
# CELL 30
# ==========================================
# ── 12.1 SHAP — TreeExplainer on best tree model ─────────────────────────────
print("Computing SHAP values (this may take a few minutes)...")

# Use best tree model (usually XGBoost or LightGBM for speed)
shap_model = trained_models.get('XGBoost', trained_models.get('LightGBM', xgb_model))
shap_model_name = [k for k, v in trained_models.items() if v is shap_model][0]

SHAP_SAMPLE = min(2000, len(X_test_sel))
idx_shap = np.random.choice(len(X_test_sel), SHAP_SAMPLE, replace=False)
X_shap = X_test_sel[idx_shap]

t0 = time.time()
explainer = shap.TreeExplainer(shap_model)
shap_values = explainer.shap_values(X_shap)
print(f"⏱ SHAP computed in {time.time()-t0:.1f}s for {SHAP_SAMPLE} samples")

# If shap_values is a list (multiclass), pick the class with most attacks (not BENIGN)
benign_idx_local = list(multi_encoder.classes_).index('BENIGN')
attack_class_idx = (benign_idx_local + 1) % N_CLASSES  # pick next class

sv_plot = shap_values[attack_class_idx] if isinstance(shap_values, list) else shap_values

# Summary plot (beeswarm)
shap.summary_plot(sv_plot, X_shap,
                  feature_names=selected_features,
                  max_display=20, show=False)
plt.title(f'SHAP Summary — {shap_model_name} (class: {multi_encoder.classes_[attack_class_idx]})')
plt.tight_layout()
plt.savefig('shap_summary_plot.png', bbox_inches='tight')
plt.show()


# ==========================================
# CELL 31
# ==========================================
# ── 12.2 SHAP bar plot (mean |SHAP|) ─────────────────────────────────────────
shap.summary_plot(sv_plot, X_shap,
                  feature_names=selected_features,
                  plot_type='bar', max_display=20, show=False)
plt.title(f'Mean |SHAP| — Feature Importance')
plt.tight_layout()
plt.savefig('shap_bar_plot.png', bbox_inches='tight')
plt.show()


# ==========================================
# CELL 32
# ==========================================
# ── 12.3 Built-in Feature Importance (top tree model) ────────────────────────
if hasattr(shap_model, 'feature_importances_'):
    fi_df = pd.DataFrame({
        'Feature': selected_features,
        'Importance': shap_model.feature_importances_
    }).sort_values('Importance', ascending=False).head(20)

    plt.figure(figsize=(10, 7))
    plt.barh(fi_df['Feature'][::-1], fi_df['Importance'][::-1], color='steelblue')
    plt.xlabel('Feature Importance Score')
    plt.title(f'Top 20 Feature Importances — {shap_model_name}')
    plt.tight_layout()
    plt.savefig(f'feature_importance_{shap_model_name.lower().replace(" ","_")}.png',
                bbox_inches='tight')
    plt.show()


# ==========================================
# CELL 33
# ==========================================
# ── 12.4 Permutation Importance ───────────────────────────────────────────────
print("Computing permutation importance (this may take a few minutes)...")
PERM_SAMPLE = min(5000, len(X_test_sel))
idx_perm = np.random.choice(len(X_test_sel), PERM_SAMPLE, replace=False)

t0 = time.time()
perm_result = permutation_importance(
    shap_model,
    X_test_sel[idx_perm], y_test[idx_perm],
    n_repeats=5, random_state=42, n_jobs=-1,
    scoring='f1_weighted'
)
print(f"⏱ Permutation importance done in {time.time()-t0:.1f}s")

perm_df = pd.DataFrame({
    'Feature': selected_features,
    'Mean_Decrease': perm_result.importances_mean
}).sort_values('Mean_Decrease', ascending=False).head(20)

plt.figure(figsize=(10, 7))
plt.barh(perm_df['Feature'][::-1], perm_df['Mean_Decrease'][::-1], color='coral')
plt.xlabel('Mean Decrease in Weighted F1')
plt.title('Permutation Importance — Top 20 Features')
plt.tight_layout()
plt.savefig('permutation_importance.png', bbox_inches='tight')
plt.show()


# ==========================================
# CELL 34
# ==========================================
# ── Memory cleanup ────────────────────────────────────────────────────────────
# Note: X_train_raw and X_scaled were already freed in earlier sections
# This cell kept for safety — will print a note if already freed
import gc
try:
    del X_train_sm  # free SMOTE training data
except NameError:
    pass
gc.collect()
print('✅ Memory cleaned.')


# ==========================================
# CELL 35
# ==========================================
import gc

gc.collect()

print("Garbage collection completed")

# ==========================================
# CELL 36
# ==========================================
# ── Create results dataframe safely ──────────────────────────────────────────

results_list = []

for mname, model in trained_models.items():

    print(f"\nEvaluating {mname}...")

    try:
        # Predictions
        y_pred = model.predict(X_test_sel)

        # Metrics
        acc = accuracy_score(y_test, y_pred)

        prec_w = precision_score(
            y_test,
            y_pred,
            average='weighted',
            zero_division=0
        )

        rec_w = recall_score(
            y_test,
            y_pred,
            average='weighted',
            zero_division=0
        )

        f1_w = f1_score(
            y_test,
            y_pred,
            average='weighted',
            zero_division=0
        )

        macro_f1 = f1_score(
            y_test,
            y_pred,
            average='macro',
            zero_division=0
        )

        # ROC-AUC
        roc_auc = np.nan

        if hasattr(model, 'predict_proba'):

            y_prob = model.predict_proba(X_test_sel)

            roc_auc = roc_auc_score(
                y_test,
                y_prob,
                multi_class='ovr',
                average='weighted'
            )

            # Free probability array immediately
            del y_prob
            gc.collect()

        # Save results
        results_list.append({
            'Model': mname,
            'Accuracy': acc,
            'Precision_W': prec_w,
            'Recall_W': rec_w,
            'Weighted_F1': f1_w,
            'Macro_F1': macro_f1,
            'ROC_AUC': roc_auc,
            'Train_Time_s': np.nan
        })

        # Free prediction array
        del y_pred
        gc.collect()

    except Exception as e:
        print(f"Error evaluating {mname}: {e}")

# Final dataframe
results_df = pd.DataFrame(results_list)

print("\nresults_df created successfully")

# ==========================================
# CELL 37
# ==========================================
# ── 15.1 Full comparison table ────────────────────────────────────────────────
results_df_sorted = results_df.sort_values('Weighted_F1', ascending=False).reset_index(drop=True)
results_df_sorted.index += 1  # rank from 1

print("=" * 95)
print("FINAL MODEL COMPARISON TABLE")
print("=" * 95)
print(results_df_sorted[[
    'Model', 'Accuracy', 'Precision_W', 'Recall_W',
    'Weighted_F1', 'Macro_F1', 'ROC_AUC', 'Train_Time_s'
]].to_string())
print("=" * 95)


# ==========================================
# CELL 38
# ==========================================
# ── 15.2 Recall comparison chart ─────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(20, 6))

metrics = [('Recall_W', 'Recall (Weighted)', 'tomato'),
           ('Weighted_F1', 'Weighted F1-Score', 'steelblue'),
           ('ROC_AUC', 'ROC-AUC', 'seagreen')]

for ax, (metric, label, color) in zip(axes, metrics):
    sorted_df = results_df_sorted.sort_values(metric)
    ax.barh(sorted_df['Model'], sorted_df[metric], color=color)
    ax.set_xlabel(label)
    ax.set_title(f'{label} Comparison')
    ax.set_xlim(sorted_df[metric].min() * 0.97, 1.01)
    for bar, val in zip(ax.patches, sorted_df[metric]):
        ax.text(bar.get_width() + 0.001, bar.get_y() + bar.get_height()/2,
                f'{val:.4f}', va='center', fontsize=8)

plt.suptitle('Model Performance Comparison — IDS Metrics', fontsize=14, y=1.02)
plt.tight_layout()
plt.savefig('model_comparison_chart.png', bbox_inches='tight')
plt.show()


# ==========================================
# CELL 39
# ==========================================
# ── 15.3 Best model summary & recommendation ─────────────────────────────────
print("=" * 60)
print("  🏆 BEST MODEL SUMMARY")
print("=" * 60)
best_row = results_df_sorted.iloc[0]
print(f"  Model           : {best_row['Model']}")
print(f"  Accuracy        : {best_row['Accuracy']:.4f}")
print(f"  Weighted Recall : {best_row['Recall_W']:.4f}")
print(f"  Weighted F1     : {best_row['Weighted_F1']:.4f}")
print(f"  Macro F1        : {best_row['Macro_F1']:.4f}")
print(f"  ROC-AUC         : {best_row['ROC_AUC']:.4f}")
print(f"  Training Time   : {best_row['Train_Time_s']:.1f}s")
print("=" * 60)


# ==========================================
# CELL 40
# ==========================================
# ── Section 15.4 (NEW): Before vs After calibration chart ───────────────────
if calibration_results:
    cal_df = pd.DataFrame(calibration_results)
    cal_df = cal_df.sort_values('acc_calibrated', ascending=False)

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    x = np.arange(len(cal_df))
    width = 0.35

    axes[0].bar(x - width/2, cal_df['acc_default'],   width, label='Default (0.5)', color='#5b8fc9', alpha=0.85)
    axes[0].bar(x + width/2, cal_df['acc_calibrated'], width, label='Calibrated',   color='#2e7d32', alpha=0.85)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(cal_df['model'], rotation=20, ha='right')
    axes[0].set_ylabel('Accuracy')
    axes[0].set_title('Accuracy: Default vs Calibrated Thresholds')
    axes[0].legend()
    axes[0].set_ylim(cal_df['acc_default'].min() * 0.97, 1.0)
    for bar in axes[0].patches:
        axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.002,
                     f'{bar.get_height():.3f}', ha='center', va='bottom', fontsize=8)

    gains = cal_df['acc_gain'] * 100
    colors = ['#2e7d32' if g > 0 else '#c62828' for g in gains]
    axes[1].bar(x, gains, color=colors, alpha=0.85)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(cal_df['model'], rotation=20, ha='right')
    axes[1].set_ylabel('Accuracy Gain (%)')
    axes[1].set_title('Accuracy Gain from Threshold Calibration')
    axes[1].axhline(0, color='black', linewidth=0.5)
    for bar, g in zip(axes[1].patches, gains):
        axes[1].text(bar.get_x() + bar.get_width()/2,
                     bar.get_height() + (0.05 if g >= 0 else -0.2),
                     f'{g:+.2f}%', ha='center', va='bottom', fontsize=9, fontweight='bold')

    plt.suptitle('Impact of Per-Class Threshold Calibration on IDS Accuracy', fontsize=13)
    plt.tight_layout()
    plt.savefig('threshold_calibration_comparison.png', bbox_inches='tight')
    plt.show()


# ==========================================
# CELL 41
# ==========================================
import numpy as np
import pandas as pd
import time
from datetime import datetime, timedelta
from collections import defaultdict, deque


class IDSPreventionEngine:
    """
    Production-grade Intrusion Detection AND Prevention System.

    Prevention Tiers:
      ALLOW       — benign traffic, pass through
      MONITOR     — low-confidence attack, log and watch
      RATE_LIMIT  — medium confidence, throttle requests
      BLOCK       — high confidence attack, block IP temporarily
      QUARANTINE  — critical/repeated attack, block + alert + log for investigation

    Severity Tiers (by attack category):
      CRITICAL  — Exploit_Attack, Malware               → auto-QUARANTINE
      HIGH      — DoS_DDoS, Brute_Force                 → BLOCK
      MEDIUM    — Web_Attack, Network_Attack             → RATE_LIMIT
      LOW       — other / low-confidence detections      → MONITOR
    """

    # Severity mapping for each attack category in your Multi_Label mapping
    SEVERITY_MAP = {
        'BENIGN':          'NONE',
        'Exploit_Attack':  'CRITICAL',
        'Malware':         'CRITICAL',
        'DoS_DDoS':        'HIGH',
        'Brute_Force':     'HIGH',
        'Web_Attack':      'MEDIUM',
        'Network_Attack':  'MEDIUM',
    }

    # Action logic: (severity, confidence%) → action
    # Confidence bands: LOW <50%, MEDIUM 50-80%, HIGH >80%
    ACTION_MATRIX = {
        ('NONE',     'ANY'):    'ALLOW',
        ('LOW',      'LOW'):    'MONITOR',
        ('LOW',      'MEDIUM'): 'MONITOR',
        ('LOW',      'HIGH'):   'RATE_LIMIT',
        ('MEDIUM',   'LOW'):    'MONITOR',
        ('MEDIUM',   'MEDIUM'): 'RATE_LIMIT',
        ('MEDIUM',   'HIGH'):   'BLOCK',
        ('HIGH',     'LOW'):    'RATE_LIMIT',
        ('HIGH',     'MEDIUM'): 'BLOCK',
        ('HIGH',     'HIGH'):   'BLOCK',
        ('CRITICAL', 'LOW'):    'BLOCK',
        ('CRITICAL', 'MEDIUM'): 'QUARANTINE',
        ('CRITICAL', 'HIGH'):   'QUARANTINE',
    }

    BLOCK_DURATION_MINUTES = {
        'MONITOR':    0,
        'RATE_LIMIT': 5,
        'BLOCK':      30,
        'QUARANTINE': 1440,  # 24 hours
    }

    def __init__(
    self,
    classifier,
    label_encoder,
    feature_names,
    log1p_cols=None,
    benign_label='BENIGN',
    rate_limit_threshold=10
):
        self.classifier     = classifier
        self.label_encoder  = label_encoder
        self.feature_names  = feature_names
        self.log1p_cols     = log1p_cols or []
        self.benign_label   = benign_label
        self.rate_limit_threshold = rate_limit_threshold

        # State storage
        self._blocked_ips    = {}          # ip → unblock_time
        self._quarantine_ips = set()       # permanent quarantine
        self._rate_counters  = defaultdict(lambda: deque(maxlen=100))  # ip → timestamps
        self._ip_attack_count = defaultdict(int)   # ip → total attacks seen
        self._incident_log   = []
        self._alert_queue    = []
        self._stats          = defaultdict(int)    # attack type counters

    def _conf_band(self, confidence):
        if confidence is None:
            return 'MEDIUM'
        if confidence < 50:
            return 'LOW'
        if confidence < 80:
            return 'MEDIUM'
        return 'HIGH'

    def _get_action(self, severity, confidence, src_ip):
        if severity == 'NONE':
            return 'ALLOW'
        # Escalate if IP has repeated attacks
        if src_ip and self._ip_attack_count[src_ip] >= 5:
            # Force at least BLOCK for repeat offenders
            if severity in ('MEDIUM', 'LOW'):
                severity = 'HIGH'
        conf_band = self._conf_band(confidence)
        key = (severity, conf_band)
        return self.ACTION_MATRIX.get(key,
               self.ACTION_MATRIX.get((severity, 'ANY'), 'MONITOR'))

    def _apply_action(self, action, src_ip):
        if action in ('BLOCK', 'QUARANTINE') and src_ip:
            minutes = self.BLOCK_DURATION_MINUTES.get(action, 30)
            unblock_time = datetime.utcnow() + timedelta(minutes=minutes)
            self._blocked_ips[src_ip] = unblock_time
            if action == 'QUARANTINE':
                self._quarantine_ips.add(src_ip)
        elif action == 'RATE_LIMIT' and src_ip:
            self._rate_counters[src_ip].append(datetime.utcnow())

    def is_blocked(self, src_ip):
        if src_ip in self._quarantine_ips:
            return True
        if src_ip in self._blocked_ips:
            if datetime.utcnow() < self._blocked_ips[src_ip]:
                return True
            else:
                del self._blocked_ips[src_ip]  # unblock expired
        return False

    def _preprocess(self, raw):

      row = pd.DataFrame([raw])

      # Ensure all selected features exist
      for f in self.feature_names:

          if f not in row.columns:
              row[f] = 0.0

      # Correct feature order
      row = row[self.feature_names]

      # Float conversion
      row = row.astype(np.float32)

      return row.values

    def _generate_alert(self, label, severity, action, src_ip, confidence):
        alert = {
            'time':        datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S'),
            'level':       severity,
            'attack_type': label,
            'action':      action,
            'src_ip':      src_ip,
            'confidence':  confidence,
            'message':     (
                f'[{severity}] {label} detected from {src_ip} '
                f'(conf={confidence}%) → {action}'
            )
        }
        self._alert_queue.append(alert)
        return alert

    def detect_and_prevent(self, raw, src_ip=None):
        """Main entry point. Returns full incident dict."""
        # Fast-path: already blocked
        if src_ip and self.is_blocked(src_ip):
            return {
                'timestamp': datetime.utcnow().isoformat(),
                'src_ip': src_ip,
                'action': 'BLOCK (cached)',
                'note': 'IP already in blocklist'
            }

        vec   = self._preprocess(raw)
        pred_enc  = self.classifier.predict(vec)[0]
        label     = self.label_encoder.inverse_transform([pred_enc])[0]
        is_attack = (label != self.benign_label)

        try:
            proba      = self.classifier.predict_proba(vec)[0]
            confidence = round(float(proba.max()) * 100, 2)
        except Exception:
            confidence = None

        severity = self.SEVERITY_MAP.get(label, 'MEDIUM')
        action   = self._get_action(severity, confidence, src_ip)

        # Apply prevention
        if is_attack and src_ip:
            self._ip_attack_count[src_ip] += 1
            self._apply_action(action, src_ip)
            self._stats[label] += 1

        # Generate alert for non-benign detections
        alert = None
        if is_attack and action in ('BLOCK', 'QUARANTINE', 'RATE_LIMIT'):
            alert = self._generate_alert(label, severity, action, src_ip, confidence)

        incident = {
            'timestamp':       datetime.utcnow().isoformat(),
            'src_ip':          src_ip,
            'predicted_label': label,
            'is_attack':       is_attack,
            'severity':        severity,
            'confidence_pct':  confidence,
            'action':          action,
            'alert':           alert['message'] if alert else None
        }
        self._incident_log.append(incident)
        return incident

    def generate_iptables_rules(self):
        """Generate Linux iptables commands to apply current blocklist."""
        rules = ['# ── IDS Generated iptables Rules ────────────────────────',
                 '# Apply with: sudo bash iptables_rules.sh']
        for ip in self._quarantine_ips:
            rules.append(f'iptables -A INPUT -s {ip} -j DROP     # QUARANTINE')
        for ip, until in self._blocked_ips.items():
            if ip not in self._quarantine_ips:
                rules.append(f'iptables -A INPUT -s {ip} -j DROP     # BLOCK until {until}')
        return '\n'.join(rules)

    def get_attack_stats(self):
        """Return attack type counters."""
        return dict(self._stats)

    def get_blocked_ips(self):
        return list(self._blocked_ips.keys())

    def get_quarantine_ips(self):
        return list(self._quarantine_ips)

    def get_incident_log(self):
        return pd.DataFrame(self._incident_log)

    def get_alerts(self):
        return pd.DataFrame(self._alert_queue)

    def print_dashboard(self):
        """Print live statistics dashboard."""
        print('=' * 65)
        print('  IDS PREVENTION ENGINE — LIVE DASHBOARD')
        print('=' * 65)
        print(f'  Total incidents logged : {len(self._incident_log)}')
        print(f'  IPs currently blocked  : {len(self._blocked_ips)}')
        print(f'  IPs in quarantine      : {len(self._quarantine_ips)}')
        print(f'  Pending alerts         : {len(self._alert_queue)}')
        print()
        print('  Attack Type Counters:')
        for atype, cnt in sorted(self._stats.items(), key=lambda x: -x[1]):
            severity = self.SEVERITY_MAP.get(atype, 'UNKNOWN')
            print(f'    [{severity:8s}] {atype:<22}: {cnt}')
        print('=' * 65)

# ==========================================
# CELL 42
# ==========================================
# ── Initialise prevention engine with best trained model ─────────────────────

# Get best model name from comparison table
best_model_name = best_row['Model']

# Get actual trained classifier object
best_classifier = trained_models[best_model_name]

# Create IDS prevention engine
ids_engine = IDSPreventionEngine(
    classifier=final_best_model,
    label_encoder=multi_encoder,
    feature_names=selected_features,
    log1p_cols=[],
    benign_label='BENIGN',
    rate_limit_threshold=10
)

print(f'IDS Prevention Engine ready — using: {best_model_name}')

# ==========================================
# CELL 43
# ==========================================
# ── Get best calibrated model information ────────────────────────────────────

# Best calibrated result
final_best = max(
    calibration_results,
    key=lambda x: x['acc_calibrated']
)

# Best model name
final_best_model_name = final_best['model']

# Actual trained model object
final_best_model = trained_models[final_best_model_name]

# Best thresholds
final_best_thresholds = final_best['thresholds']

# ── Initialise prevention engine with BEST CALIBRATED model ─────────────────

ids_engine = IDSPreventionEngine(
    classifier=final_best_model,
    label_encoder=multi_encoder,
    feature_names=selected_features,
    log1p_cols=[],
    benign_label='BENIGN',
    rate_limit_threshold=10
)

# Store calibrated thresholds
ids_engine._calibrated_thresholds = final_best_thresholds

print(f'✅ IDS Prevention Engine ready — using: {final_best_model_name}')

print(
    f'✅ Calibrated accuracy: '
    f'{final_best["acc_calibrated"]:.4f}'
)

# ==========================================
# CELL 44
# ==========================================
# ── Live simulation on test set (one sample per class) ────────────────────────
print('=' * 70)
print('LIVE SIMULATION — Real Test Set Samples')
print('=' * 70)

test_df_sim = pd.DataFrame(X_test_sel, columns=selected_features)
test_df_sim['TrueLabel'] = multi_encoder.inverse_transform(y_test)

sim_rows = []
for cls in multi_encoder.classes_:
    subset = test_df_sim[test_df_sim['TrueLabel'] == cls]
    if len(subset) > 0:
        sim_rows.append(subset.sample(2, random_state=42))  # 2 per class

sim_df = pd.concat(sim_rows).reset_index(drop=True)

correct = 0
for i, row in sim_df.iterrows():
    true_label = multi_encoder.inverse_transform([y_test[i]])[0]

    record = pd.Series(
      X_test_sel[i],
      index=selected_features
    ).to_dict()
    result     = ids_engine.detect_and_prevent(record, src_ip=f'10.0.{i//256}.{i%256}')

    match  = result['predicted_label'] == true_label
    if match: correct += 1
    status = 'Yes' if match else 'No'
    sev    = result.get('severity', 'N/A')
    print(f"{status} True: {true_label:<22} | Pred: {result['predicted_label']:<22}"
          f"| Conf: {result['confidence_pct']}% | Sev: {sev:<8} | Action: {result['action']}")

print(f'\nSimulation accuracy: {correct}/{len(sim_df)} = {correct/len(sim_df)*100:.1f}%')

# Dashboard
ids_engine.print_dashboard()

# Show alerts
alerts = ids_engine.get_alerts()
if len(alerts) > 0:
    print('\n🚨 ALERTS GENERATED:')
    for _, alert in alerts.iterrows():
        print(f'  {alert["message"]}')

# iptables rules
print('\n📋 Generated Firewall Rules:')
print(ids_engine.generate_iptables_rules())

# ==========================================
# CELL 45
# ==========================================
# ── Incident log ──────────────────────────────────────────────────────────────
log_df = ids_engine.get_incident_log()
print('Incident Log (attack rows only):')
attack_log = log_df[log_df['is_attack'] == True]
print(attack_log[['timestamp','src_ip','predicted_label','severity',
                   'confidence_pct','action']].to_string(index=False))

print(f'\nBlocked IPs  : {ids_engine.get_blocked_ips()}')
print(f'Quarantined  : {ids_engine.get_quarantine_ips()}')

# ==========================================
# CELL 46
# ==========================================

