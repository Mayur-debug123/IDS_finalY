import os
import sys
import time
import json
import gc
import pickle
import joblib
import pandas as pd
import numpy as np
from collections import Counter

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, RobustScaler
from sklearn.feature_selection import VarianceThreshold, SelectKBest, mutual_info_classif
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             roc_auc_score, confusion_matrix, classification_report,
                             roc_curve, precision_recall_curve, average_precision_score)
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from imblearn.over_sampling import SMOTE
import shap

def run_pipeline():
    print("=" * 60)
    print("STARTING MACHINE LEARNING PIPELINE TRAINING")
    print("=" * 60)
    
    dataset_path = r"C:\Users\Admin\Downloads\od-ids2022\OD-IDS2022-Dataset.csv"
    if not os.path.exists(dataset_path):
        print(f"Error: Dataset not found at {dataset_path}")
        sys.exit(1)
        
    t_start = time.time()
    
    # ── 1. Load Data ──────────────────────────────────────────────────────────
    print("\n[1/9] Loading dataset...")
    t0 = time.time()
    raw_traffic_df = pd.read_csv(dataset_path, low_memory=False)
    print(f"   Loaded {raw_traffic_df.shape[0]:,} rows × {raw_traffic_df.shape[1]} columns in {time.time()-t0:.1f}s")
    
    # ── 2. Data Cleaning & Downcasting ────────────────────────────────────────
    print("\n[2/9] Cleaning data and optimizing memory...")
    t0 = time.time()
    traffic_df = raw_traffic_df.copy()
    del raw_traffic_df
    gc.collect()
    
    traffic_df.columns = traffic_df.columns.str.strip()
    
    # Fill missing values
    num_cols = traffic_df.select_dtypes(include=[np.number]).columns
    cat_cols = traffic_df.select_dtypes(include='object').columns
    
    traffic_df[num_cols] = traffic_df[num_cols].fillna(traffic_df[num_cols].median())
    for col in cat_cols:
        if col != 'Label':
            mode = traffic_df[col].mode()
            if not mode.empty:
                traffic_df[col] = traffic_df[col].fillna(mode[0])
                
    # Remove duplicates
    before = len(traffic_df)
    traffic_df.drop_duplicates(inplace=True)
    print(f"   Removed {before - len(traffic_df):,} duplicate rows.")
    
    # Drop identifiers
    id_cols = [c for c in ['Src IP', 'Dst IP', 'Src Port', 'Dst Port'] if c in traffic_df.columns]
    traffic_df.drop(columns=id_cols, inplace=True)
    
    # Infinite values
    num_cols = traffic_df.select_dtypes(include=[np.number]).columns
    traffic_df[num_cols] = traffic_df[num_cols].replace([np.inf, -np.inf], np.nan)
    for col in num_cols:
        cap = traffic_df[col].quantile(0.99)
        traffic_df[col].fillna(cap, inplace=True)
        
    # Reduce memory usage utility
    def reduce_mem_usage(df: pd.DataFrame) -> pd.DataFrame:
        for col in df.select_dtypes(include=[np.number]).columns:
            col_min, col_max = df[col].min(), df[col].max()
            if df[col].dtype.kind == 'f':
                df[col] = df[col].astype(np.float32)
            elif df[col].dtype.kind == 'i':
                if col_min >= np.iinfo(np.int32).min and col_max <= np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
        return df
        
    traffic_df = reduce_mem_usage(traffic_df)
    print(f"   Memory downcasted. Dataset shape: {traffic_df.shape[0]:,} rows × {traffic_df.shape[1]} columns in {time.time()-t0:.1f}s")
    
    # ── 3. Label Encoding & Mapping ───────────────────────────────────────────
    print("\n[3/9] Mapping labels and encoding classes...")
    def map_attack(label):
        if label == "BENIGN":
            return "BENIGN"
        elif label in ["SYN Floods", "Denial-of-service", "Distributed_denial-of-service", "Slow_HTTP_attack"]:
            return "DoS_DDoS"
        elif label in ["ARP_Spoofing", "Man-in-the-middle", "Fragmented Packet Attacks", "TCP_Session_Hijacking"]:
            return "Network_Attack"
        elif label in ["Time-based SQL Injection", "Persistent Cross-Site Scripting in Blog page", "ManageEngine ADSelfService Plus 6.1 - CSV Injection"]:
            return "Web_Attack"
        elif label in ["Brute Force Attacks"]:
            return "Brute_Force"
        elif label in ["Ransomware (Malware)", "DLL Hijacking", "EXE Hijacking", "EXE HijackinPrintNightMare-RCE", "Firmware Vulnerabilitie"]:
            return "Malware"
        elif label in ["Apache_flink_directory_traversal", "Authenticated Remote Code Execution", "Exploiting Node Deserialization", 
                       "Google Chrome Remote Code Execution via Browser", "Kernel Exploitation", "Print Spooler Service - Local Privilege Escalation", 
                       "Privilege Escalation Using Unquoted Service Path", "Remote Code Execution via Unrestricted File Upload access", 
                       "Unauthenticated Arbitrary File Upload", "Unauthenticated RCE in Credit Card Customer Care System", 
                       "Webmin 1.962 - Package Update Escape Bypass RCE"]:
            return "Exploit_Attack"
        else:
            return "Other_Attack"
            
    traffic_df['Multi_Label'] = traffic_df['Label'].map(map_attack).fillna("Other_Attack")
    traffic_df = traffic_df[traffic_df['Multi_Label'] != "Other_Attack"].copy()
    
    multi_encoder = LabelEncoder()
    traffic_df['Multi_Encoded'] = multi_encoder.fit_transform(traffic_df['Multi_Label'])
    
    print("   Mapped classes & counts:")
    for idx, cls in enumerate(multi_encoder.classes_):
        cnt = (traffic_df['Multi_Encoded'] == idx).sum()
        print(f"     {idx} -> {cls:<16}: {cnt:>7,} samples")
        
    # Save the label encoder
    joblib.dump(multi_encoder, 'label_encoder.pkl')
    print("   Saved 'label_encoder.pkl'")
    
    # ── 4. Skewness Handling & Feature Engineering ────────────────────────────
    print("\n[4/9] Skewness handling and feature engineering...")
    t0 = time.time()
    
    EXCLUDE_COLS = ['Label', 'EncodedLabel', 'Multi_Label', 'Multi_Encoded']
    feature_cols = [c for c in traffic_df.select_dtypes(include=np.number).columns if c not in EXCLUDE_COLS]
    
    # Identify skewed features
    BINARY_COLS = [c for c in feature_cols if traffic_df[c].nunique() <= 2]
    TRANSFORM_CANDIDATES = [c for c in feature_cols if c not in BINARY_COLS]
    raw_skewness = traffic_df[TRANSFORM_CANDIDATES].skew()
    
    log1p_cols = [col for col in TRANSFORM_CANDIDATES if raw_skewness.get(col, 0) > 1.0 and traffic_df[col].min() >= 0]
    
    # Log transform
    traffic_df_transformed = traffic_df.copy()
    for col in log1p_cols:
        traffic_df_transformed[col] = np.log1p(traffic_df_transformed[col].clip(lower=0))
        
    # Attack specific feature helper
    def add_attack_specific_features(df, feature_names):
        df = df.copy()
        
        bps_col = next((c for c in feature_names if 'Byts/s' in c or 'Bytes/s' in c), None)
        pps_col = next((c for c in feature_names if 'Pkts/s' in c or 'Packets/s' in c), None)
        if bps_col and pps_col:
            df['Bytes_Per_Packet'] = (df[bps_col] / (df[pps_col] + 1e-6)).astype(np.float32)
            
        fwd_col = next((c for c in feature_names if 'Fwd Pkt Len Mean' in c), None)
        bwd_col = next((c for c in feature_names if 'Bwd Pkt Len Mean' in c), None)
        if fwd_col and bwd_col:
            df['Fwd_Bwd_Ratio'] = (df[fwd_col] / (df[bwd_col] + 1e-6)).astype(np.float32)
            
        dur_col = next((c for c in feature_names if 'Flow Duration' in c), None)
        if dur_col and pps_col:
            df['Duration_Per_Pkt'] = (df[dur_col] / (df[pps_col] + 1e-6)).astype(np.float32)
            df['Connection_Burst'] = ((df[dur_col] < 100000) & (df[pps_col] > 100)).astype(np.float32)
            
        size_cols = [c for c in feature_names if 'Pkt Len' in c and 'Mean' in c]
        if len(size_cols) >= 2:
            df['Packet_Size_Variance'] = df[size_cols].var(axis=1).astype(np.float32)
            
        flag_cols = [c for c in feature_names if 'Flag' in c]
        if flag_cols:
            df['Flag_Anomaly'] = df[flag_cols].sum(axis=1).astype(np.float32)
            
        empty_pkt_col = next((c for c in feature_names if 'Min' in c and 'Pkt Len' in c), None)
        if empty_pkt_col:
            df['Empty_Packet_Ratio'] = (df[empty_pkt_col] == 0).astype(np.float32)
            
        rst_col = next((c for c in feature_names if 'RST Flag Cnt' in c), None)
        if rst_col and pps_col:
            df['RST_Ratio'] = (df[rst_col] / (df[pps_col] + 1e-6)).astype(np.float32)
            
        return df
        
    feature_names_raw = [c for c in traffic_df_transformed.select_dtypes(include=np.number).columns if c not in EXCLUDE_COLS]
    traffic_df_transformed = add_attack_specific_features(traffic_df_transformed, feature_names_raw)
    
    # Feature matrix & target vector
    target_col = 'Multi_Encoded'
    feature_matrix = traffic_df_transformed.drop(columns=['Label', 'Multi_Label', 'Multi_Encoded'], errors='ignore')
    feature_matrix = feature_matrix.select_dtypes(include=[np.number])
    target_vector = traffic_df_transformed[target_col]
    
    del traffic_df, traffic_df_transformed
    gc.collect()
    
    print(f"   Engineered features added. Feature matrix shape: {feature_matrix.shape} in {time.time()-t0:.1f}s")
    
    # ── 5. Train-Test Split & Scaling ──────────────────────────────────────────
    print("\n[5/9] Splitting dataset & scaling features...")
    t0 = time.time()
    
    feature_cols_clean = feature_matrix.columns.tolist()
    X_raw = feature_matrix.values.astype(np.float32)
    y_raw = target_vector.values
    
    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        X_raw, y_raw, test_size=0.2, random_state=42, stratify=y_raw
    )
    
    scaler = RobustScaler()
    X_train_scaled = scaler.fit_transform(X_train_raw).astype(np.float32)
    X_test_scaled = scaler.transform(X_test_raw).astype(np.float32)
    
    # Outlier clipping
    CLIP_LIMIT = 10.0
    X_train_scaled = np.clip(X_train_scaled, -CLIP_LIMIT, CLIP_LIMIT)
    X_test_scaled = np.clip(X_test_scaled, -CLIP_LIMIT, CLIP_LIMIT)
    
    joblib.dump(scaler, 'scaler.pkl')
    print("   Saved 'scaler.pkl'")
    print(f"   Train set shape: {X_train_scaled.shape}, Test set shape: {X_test_scaled.shape}")
    
    del X_raw, X_train_raw, X_test_raw
    gc.collect()
    
    # ── 6. Feature Selection ──────────────────────────────────────────────────
    print("\n[6/9] Selecting features using Mutual Information...")
    t0 = time.time()
    
    # Step 1: VarianceThreshold
    vt = VarianceThreshold(threshold=0.0)
    X_train_vt = vt.fit_transform(X_train_scaled)
    X_test_vt = vt.transform(X_test_scaled)
    feature_names_vt = [feature_cols_clean[i] for i in vt.get_support(indices=True)]
    
    # Step 2: Correlation filter (>0.95)
    MI_SAMPLE = min(50000, len(X_train_vt))
    idx_mi = np.random.choice(len(X_train_vt), MI_SAMPLE, replace=False)
    corr_df = pd.DataFrame(X_train_vt[idx_mi], columns=feature_names_vt)
    corr_mat = corr_df.corr().abs()
    upper_tri = corr_mat.where(np.triu(np.ones_like(corr_mat, dtype=bool), k=1))
    drop_cols = [c for c in upper_tri.columns if any(upper_tri[c] > 0.95)]
    keep_idx = [i for i, n in enumerate(feature_names_vt) if n not in drop_cols]
    
    X_train_corr = X_train_vt[:, keep_idx]
    X_test_corr = X_test_vt[:, keep_idx]
    feature_names_corr = [feature_names_vt[i] for i in keep_idx]
    
    # Step 3: Mutual Information Selection
    K_BEST = 40
    selector = SelectKBest(score_func=mutual_info_classif, k=min(K_BEST, X_train_corr.shape[1]))
    selector.fit(X_train_corr[idx_mi[:min(MI_SAMPLE, len(X_train_corr))]],
                 y_train[idx_mi[:min(MI_SAMPLE, len(X_train_corr))]])
                 
    X_train_sel = selector.transform(X_train_corr).astype(np.float32)
    X_test_sel = selector.transform(X_test_corr).astype(np.float32)
    
    feat_scores = pd.Series(selector.scores_, index=feature_names_corr)
    selected_features = feat_scores.nlargest(K_BEST).index.tolist()
    
    joblib.dump(selected_features, 'selected_features.pkl')
    print("   Saved 'selected_features.pkl'")
    print(f"   Selected {len(selected_features)} features. Top 5: {selected_features[:5]}")
    
    # Save feature scores & importances for UI visualization
    mi_importances = feat_scores.nlargest(K_BEST).to_dict()
    
    del X_train_corr, X_test_corr, X_train_vt, X_test_vt, corr_df
    gc.collect()
    
    # ── 7. Oversampling (SMOTE) ───────────────────────────────────────────────
    print("\n[7/9] Applying SMOTE (Safe strategy: Web_Attack only)...")
    t0 = time.time()
    
    class_counts_before = Counter(y_train)
    web_attack_idx = list(multi_encoder.classes_).index('Web_Attack')
    web_attack_count = class_counts_before[web_attack_idx]
    REASONABLE_TARGET = min(25000, int(web_attack_count * 1.5))
    
    sampling_strategy_safe = {web_attack_idx: int(REASONABLE_TARGET)}
    
    oversampler = SMOTE(sampling_strategy=sampling_strategy_safe, 
                        k_neighbors=min(5, min(class_counts_before.values())-1), 
                        random_state=42)
                        
    X_train_sm, y_train_sm = oversampler.fit_resample(X_train_sel, y_train)
    X_train_sm = X_train_sm.astype(np.float32)
    print(f"   SMOTE complete. Train shape: {X_train_sm.shape} in {time.time()-t0:.1f}s")
    
    # ── 8. Model Training & Threshold Calibration ──────────────────────────────
    print("\n[8/9] Training all 4 classifiers & calibrating thresholds...")
    
    # Pre-compute SHAP values on a subset (200 test set samples)
    print("   Setting up SHAP subset of 200 test rows...")
    SHAP_SAMPLE = 200
    np.random.seed(42)
    idx_shap = np.random.choice(len(X_test_sel), SHAP_SAMPLE, replace=False)
    X_shap = X_test_sel[idx_shap]
    y_shap = y_test[idx_shap]
    
    models = {
        'HistGradientBoosting': HistGradientBoostingClassifier(
            max_iter=550,
            learning_rate=0.1,
            max_leaf_nodes=64,
            max_depth=8,
            min_samples_leaf=50,
            l2_regularization=1.0,
            class_weight='balanced',
            early_stopping=True,
            n_iter_no_change=25,
            validation_fraction=0.2,
            random_state=42
        ),
        'XGBoost': XGBClassifier(
            n_estimators=500,
            learning_rate=0.05,
            max_depth=7,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=1.0,
            reg_lambda=1.0,
            eval_metric='mlogloss',
            tree_method='hist',
            n_jobs=-1,
            random_state=42
        ),
        'LightGBM': LGBMClassifier(
            n_estimators=450,
            learning_rate=0.05,
            num_leaves=83,
            max_depth=6,
            min_child_samples=50,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.5,
            reg_lambda=1.0,
            class_weight='balanced',
            n_jobs=-1,
            random_state=42,
            verbose=-1
        ),
        'Random Forest': RandomForestClassifier(
            n_estimators=250,
            max_depth=25,
            min_samples_split=4,
            min_samples_leaf=2,
            max_features='sqrt',
            class_weight='balanced_subsample',
            n_jobs=-1,
            random_state=42
        )
    }
    
    models_metadata = {}
    comparison_table = []
    
    for name, model in models.items():
        print(f"\n   Training {name} classifier...")
        t_model = time.time()
        model.fit(X_train_sm, y_train_sm)
        fit_time = time.time() - t_model
        print(f"   {name} training complete in {fit_time:.1f}s")
        
        # Threshold Calibration
        print(f"   Running per-class threshold calibration for {name}...")
        try:
            proba_test = model.predict_proba(X_test_sel)
        except Exception as e:
            print(f"   Error: Calibration failed {e}")
            proba_test = None
            
        thresholds = {}
        if proba_test is not None:
            for i, cls in enumerate(multi_encoder.classes_):
                y_bin_cls = (y_test == i).astype(int)
                prec_arr, rec_arr, thresh_arr = precision_recall_curve(y_bin_cls, proba_test[:, i])
                with np.errstate(divide='ignore', invalid='ignore'):
                    f1_arr = np.where((prec_arr + rec_arr) > 0, 2 * prec_arr * rec_arr / (prec_arr + rec_arr), 0)
                best_idx = np.argmax(f1_arr[:-1])
                thresholds[i] = float(thresh_arr[best_idx])
                print(f"     {cls:<16}: threshold={thresholds[i]:.3f} (Max F1={f1_arr[best_idx]:.4f})")
        else:
            thresholds = {i: 0.5 for i in range(len(multi_encoder.classes_))}
            
        # Save model and thresholds
        model_filename = f"saved_model_{name.replace(' ', '_')}.pkl"
        thresholds_filename = f"thresholds_{name.replace(' ', '_')}.pkl"
        joblib.dump(model, model_filename)
        joblib.dump(thresholds, thresholds_filename)
        print(f"   Saved '{model_filename}' & '{thresholds_filename}'")
        
        # Compute metrics
        y_pred_default = model.predict(X_test_sel)
        
        # Predict with thresholds
        thresh_arr = np.array([thresholds[i] for i in range(len(multi_encoder.classes_))])
        adjusted = proba_test / (thresh_arr + 1e-9)
        y_pred_calibrated = np.argmax(adjusted, axis=1)
        
        acc_default = accuracy_score(y_test, y_pred_default)
        acc_calibrated = accuracy_score(y_test, y_pred_calibrated)
        f1_default = f1_score(y_test, y_pred_default, average='weighted', zero_division=0)
        f1_calibrated = f1_score(y_test, y_pred_calibrated, average='weighted', zero_division=0)
        
        cm = confusion_matrix(y_test, y_pred_calibrated)
        cm_norm = cm.astype('float') / cm.sum(axis=1, keepdims=True)
        
        # ROC curves data
        roc_data = {}
        y_bin = np.eye(len(multi_encoder.classes_))[y_test]
        for i, cls in enumerate(multi_encoder.classes_):
            fpr, tpr, _ = roc_curve(y_bin[:, i], proba_test[:, i])
            auc_score = roc_auc_score(y_bin[:, i], proba_test[:, i])
            sample_step = max(1, len(fpr) // 200)
            roc_data[cls] = {
                'fpr': fpr[::sample_step].tolist(),
                'tpr': tpr[::sample_step].tolist(),
                'auc': float(auc_score)
            }
            
        # Precision-Recall curves data
        pr_data = {}
        for i, cls in enumerate(multi_encoder.classes_):
            prec_c, rec_c, _ = precision_recall_curve(y_bin[:, i], proba_test[:, i])
            ap_score = average_precision_score(y_bin[:, i], proba_test[:, i])
            sample_step = max(1, len(prec_c) // 200)
            pr_data[cls] = {
                'precision': prec_c[::sample_step].tolist(),
                'recall': rec_c[::sample_step].tolist(),
                'ap': float(ap_score)
            }
            
        # Pre-compute SHAP values on a subset (200 test set samples)
        print(f"   Pre-computing SHAP summary values for {name} (subset of 200 test rows)...")
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_shap)
        shap_is_list = isinstance(shap_values, list)
        
        # Save model metadata
        models_metadata[name] = {
            'acc_default': float(acc_default),
            'acc_calibrated': float(acc_calibrated),
            'f1_default': float(f1_default),
            'f1_calibrated': float(f1_calibrated),
            'confusion_matrix': cm.tolist(),
            'confusion_matrix_norm': cm_norm.tolist(),
            'roc_curves': roc_data,
            'pr_curves': pr_data,
            'shap_values_raw': [sv.tolist() for sv in shap_values] if shap_is_list else shap_values.tolist(),
            'shap_is_list': shap_is_list
        }
        
        # Add to comparison table
        comparison_table.append({
            'Model': name,
            'Accuracy': float(acc_calibrated),
            'Precision_W': float(precision_score(y_test, y_pred_calibrated, average='weighted', zero_division=0)),
            'Recall_W': float(recall_score(y_test, y_pred_calibrated, average='weighted', zero_division=0)),
            'Weighted_F1': float(f1_calibrated),
            'Macro_F1': float(f1_score(y_test, y_pred_calibrated, average='macro', zero_division=0)),
            'ROC_AUC': float(roc_auc_score(y_bin, proba_test, multi_class='ovr', average='weighted')),
            'Train_Time_s': float(fit_time)
        })
        
    # Save default models (HistGradientBoosting) as saved_model.pkl and thresholds.pkl for backwards compatibility
    import shutil
    shutil.copyfile("saved_model_HistGradientBoosting.pkl", "saved_model.pkl")
    shutil.copyfile("thresholds_HistGradientBoosting.pkl", "thresholds.pkl")
    print("   Created default copies for compatibility ('saved_model.pkl', 'thresholds.pkl').")

    # ── 9. Save Metadata ──────────────────────────────────────────────────────
    print("\n[9/9] Saving metadata metrics and assets...")
    
    metadata = {
        'log1p_cols': log1p_cols,
        'selected_features': selected_features,
        'mi_importances': mi_importances,
        'comparison_table': comparison_table,
        'shap_test_subset': X_shap.tolist(),
        'shap_test_labels': y_shap.tolist(),
        'models_metadata': models_metadata
    }
    
    with open('metrics_metadata.pkl', 'wb') as f:
        pickle.dump(metadata, f, protocol=pickle.HIGHEST_PROTOCOL)
        
    print("   Saved 'metrics_metadata.pkl' containing visual assets & static metrics for all models.")
    
    duration = time.time() - t_start
    print(f"\n[SUCCESS] PIPELINE COMPLETED SUCCESSFULLY IN {duration:.1f}s")
    print("=" * 60)

if __name__ == "__main__":
    run_pipeline()
