import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

# 1. Load the Datasets
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')
sample_sub = pd.read_csv('sample_submission.csv')

TARGET_COL = 'exit_status'
ID_COL = sample_sub.columns[0]

# 2. Advanced Feature Engineering with Duplicate-Safe Binning
def engineer_features(df):
    df = df.copy()
    # Financial interaction features
    df['balance_salary_ratio'] = df['acc_balance'] / (df['estimated_salary'] + 1e-5)
    df['tenure_age_ratio'] = df['tenure'] / (df['age'] + 1e-5)
    df['credit_by_age'] = df['credit_score'] * df['age']
    df['products_per_tenure'] = df['prod_count'] / (df['tenure'] + 1e-5)
    df['wealth_index'] = df['acc_balance'] * df['estimated_salary']
    df['is_zero_balance'] = (df['acc_balance'] == 0).astype(int)
    
    # Safe qcut handling duplicate bin edges
    try:
        df['credit_tier'] = pd.qcut(df['credit_score'], q=4, labels=False, duplicates='drop') + 1
    except Exception:
        df['credit_tier'] = 1
        
    return df

train_fe = engineer_features(train)
test_fe = engineer_features(test)

# Drop unnecessary identifier columns
cols_to_drop = ['record_id', 'customer_id', 'last_name']
X = train_fe.drop(columns=cols_to_drop + [TARGET_COL], errors='ignore')
y = train_fe[TARGET_COL]
X_test = test_fe.drop(columns=cols_to_drop, errors='ignore')

# 3. Categorical Encoding (One-Hot Encoding)
X = pd.get_dummies(X, columns=['country', 'gender'], drop_first=True)
X_test = pd.get_dummies(X_test, columns=['country', 'gender'], drop_first=True)
X, X_test = X.align(X_test, join='left', axis=1, fill_value=0)

# Impute missing values with median
X = X.fillna(X.median(numeric_only=True))
X_test = X_test.fillna(X_test.median(numeric_only=True))

# 4. Stratified 10-Fold Cross-Validation for Maximum Stability
skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

oof_xgb = np.zeros(len(train))
oof_lgb = np.zeros(len(train))
oof_rf = np.zeros(len(train))

test_xgb = np.zeros(len(test))
test_lgb = np.zeros(len(test))
test_rf = np.zeros(len(test))

print("Training high-precision ensemble models (XGBoost, LightGBM, Random Forest)...")
for fold, (train_idx, val_idx) in enumerate(skf.split(X, y), start=1):
    X_tr, y_tr = X.iloc[train_idx], y.iloc[train_idx]
    X_va, y_va = X.iloc[val_idx], y.iloc[val_idx]
    
    # Model 1: XGBoost
    xgb_model = XGBClassifier(
        n_estimators=800,
        learning_rate=0.015,
        max_depth=5,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=3,
        gamma=0.1,
        random_state=42,
        eval_metric='logloss'
    )
    xgb_model.fit(X_tr, y_tr)
    oof_xgb[val_idx] = xgb_model.predict_proba(X_va)[:, 1]
    test_xgb += xgb_model.predict_proba(X_test)[:, 1] / skf.n_splits

    # Model 2: LightGBM
    lgb_model = LGBMClassifier(
        n_estimators=800,
        learning_rate=0.015,
        max_depth=5,
        num_leaves=31,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        verbose=-1
    )
    lgb_model.fit(X_tr, y_tr)
    oof_lgb[val_idx] = lgb_model.predict_proba(X_va)[:, 1]
    test_lgb += lgb_model.predict_proba(X_test)[:, 1] / skf.n_splits

    # Model 3: Random Forest
    rf_model = RandomForestClassifier(
        n_estimators=400,
        max_depth=12,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1
    )
    rf_model.fit(X_tr, y_tr)
    oof_rf[val_idx] = rf_model.predict_proba(X_va)[:, 1]
    test_rf += rf_model.predict_proba(X_test)[:, 1] / skf.n_splits

# 5. Weighted Blending of Out-of-Fold Probabilities
oof_blend = 0.45 * oof_xgb + 0.45 * oof_lgb + 0.10 * oof_rf
test_blend = 0.45 * test_xgb + 0.45 * test_lgb + 0.10 * test_rf

# 6. Precise Decision Threshold Optimization to Maximize Macro F1-Score
best_thresh = 0.5
best_f1 = 0.0

for thresh in np.arange(0.1, 0.9, 0.005):
    preds_t = (oof_blend >= thresh).astype(int)
    score = f1_score(y, preds_t, average='macro')
    if score > best_f1:
        best_f1 = score
        best_thresh = thresh

print(f"Optimal Decision Threshold: {best_thresh:.3f}")
print(f"Optimized Ensemble Out-of-Fold Macro F1 Score: {best_f1:.5f}")

# 7. Generate Final Submission File
sample_sub[TARGET_COL] = (test_blend >= best_thresh).astype(int)
sample_sub.to_csv('submission.csv', index=False)
print("submission.csv successfully generated with high-accuracy ensemble predictions!")