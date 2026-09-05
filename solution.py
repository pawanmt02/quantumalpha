import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

# 1. Load Datasets
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')
sample_sub = pd.read_csv('sample_submission.csv')

TARGET_COL = 'exit_status'
ID_COL = sample_sub.columns[0]
cols_to_drop = ['record_id', 'customer_id', 'last_name']

# Base Preprocessing
def base_preprocess(tr, te):
    X_tr = tr.drop(columns=cols_to_drop + [TARGET_COL], errors='ignore')
    y_tr = tr[TARGET_COL]
    X_te = te.drop(columns=cols_to_drop, errors='ignore')
    
    X_tr = pd.get_dummies(X_tr, columns=['country', 'gender'], drop_first=True)
    X_te = pd.get_dummies(X_te, columns=['country', 'gender'], drop_first=True)
    X_tr, X_te = X_tr.align(X_te, join='left', axis=1, fill_value=0)
    
    X_tr = X_tr.fillna(X_tr.median(numeric_only=True))
    X_te = X_te.fillna(X_te.median(numeric_only=True))
    return X_tr, y_tr, X_te

# Feature Engineered Preprocessing
def advanced_preprocess(tr, te):
    tr_fe = tr.copy()
    te_fe = te.copy()
    for df in [tr_fe, te_fe]:
        df['balance_salary_ratio'] = df['acc_balance'] / (df['estimated_salary'] + 1e-5)
        df['tenure_age_ratio'] = df['tenure'] / (df['age'] + 1e-5)
        df['credit_by_age'] = df['credit_score'] * df['age']
        df['wealth_index'] = df['acc_balance'] * df['estimated_salary']
        df['is_zero_balance'] = (df['acc_balance'] == 0).astype(int)
        
    X_tr = tr_fe.drop(columns=cols_to_drop + [TARGET_COL], errors='ignore')
    y_tr = tr_fe[TARGET_COL]
    X_te = te_fe.drop(columns=cols_to_drop, errors='ignore')
    
    X_tr = pd.get_dummies(X_tr, columns=['country', 'gender'], drop_first=True)
    X_te = pd.get_dummies(X_te, columns=['country', 'gender'], drop_first=True)
    X_tr, X_te = X_tr.align(X_te, join='left', axis=1, fill_value=0)
    
    X_tr = X_tr.fillna(X_tr.median(numeric_only=True))
    X_te = X_te.fillna(X_te.median(numeric_only=True))
    return X_tr, y_tr, X_te

skf = StratifiedKStr = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# ==========================================
# SOLUTION 1: Standard Random Forest
# ==========================================
print("\n--- Generating Solution 1: Random Forest ---")
X1, y1, X1_test = base_preprocess(train, test)
preds_1 = np.zeros(len(test))
for train_idx, val_idx in skf.split(X1, y1):
    model = RandomForestClassifier(n_estimators=250, max_depth=10, random_state=42, n_jobs=-1)
    model.fit(X1.iloc[train_idx], y1.iloc[train_idx])
    preds_1 += model.predict_proba(X1_test)[:, 1] / 5

sub1 = sample_sub.copy()
sub1[TARGET_COL] = (preds_1 >= 0.5).astype(int)
sub1.to_csv('submission_1.csv', index=False)

# ==========================================
# SOLUTION 2: Feature-Engineered XGBoost
# ==========================================
print("--- Generating Solution 2: XGBoost with Ratios ---")
X2, y2, X2_test = advanced_preprocess(train, test)
preds_2 = np.zeros(len(test))
for train_idx, val_idx in skf.split(X2, y2):
    model = XGBClassifier(n_estimators=500, learning_rate=0.03, max_depth=5, subsample=0.8, random_state=42, eval_metric='logloss')
    model.fit(X2.iloc[train_idx], y2.iloc[train_idx])
    preds_2 += model.predict_proba(X2_test)[:, 1] / 5

sub2 = sample_sub.copy()
sub2[TARGET_COL] = (preds_2 >= 0.5).astype(int)
sub2.to_csv('submission_2.csv', index=False)

# ==========================================
# SOLUTION 3: Deep LightGBM
# ==========================================
print("--- Generating Solution 3: LightGBM ---")
X3, y3, X3_test = advanced_preprocess(train, test)
preds_3 = np.zeros(len(test))
for train_idx, val_idx in skf.split(X3, y3):
    model = LGBMClassifier(n_estimators=600, learning_rate=0.02, max_depth=6, num_leaves=31, random_state=42, verbose=-1)
    model.fit(X3.iloc[train_idx], y3.iloc[train_idx])
    preds_3 += model.predict_proba(X3_test)[:, 1] / 5

sub3 = sample_sub.copy()
sub3[TARGET_COL] = (preds_3 >= 0.5).astype(int)
sub3.to_csv('submission_3.csv', index=False)

# ==========================================
# SOLUTION 4: Extra Trees Classifier
# ==========================================
print("--- Generating Solution 4: Extra Trees ---")
X4, y4, X4_test = advanced_preprocess(train, test)
preds_4 = np.zeros(len(test))
for train_idx, val_idx in skf.split(X4, y4):
    model = ExtraTreesClassifier(n_estimators=400, max_depth=12, min_samples_split=4, random_state=42, n_jobs=-1)
    model.fit(X4.iloc[train_idx], y4.iloc[train_idx])
    preds_4 += model.predict_proba(X4_test)[:, 1] / 5

sub4 = sample_sub.copy()
sub4[TARGET_COL] = (preds_4 >= 0.5).astype(int)
sub4.to_csv('submission_4.csv', index=False)

# ==========================================
# SOLUTION 5: Weighted Ensemble Blend
# ==========================================
print("--- Generating Solution 5: Blended Ensemble ---")
preds_blend = 0.4 * preds_2 + 0.4 * preds_3 + 0.2 * preds_4
sub5 = sample_sub.copy()
sub5[TARGET_COL] = (preds_blend >= 0.5).astype(int)
sub5.to_csv('submission_5.csv', index=False)

print("\n✅ All 5 distinct solution files created successfully:")
print(" - submission_1.csv (Random Forest)")
print(" - submission_2.csv (XGBoost)")
print(" - submission_3.csv (LightGBM)")
print(" - submission_4.csv (Extra Trees)")
print(" - submission_5.csv (Blended Ensemble)")