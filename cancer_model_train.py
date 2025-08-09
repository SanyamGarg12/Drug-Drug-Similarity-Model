import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.model_selection import train_test_split, RandomizedSearchCV, StratifiedKFold, cross_val_score
from sklearn.feature_selection import mutual_info_classif, RFE, SelectKBest, f_classif, VarianceThreshold
from sklearn.linear_model import LogisticRegression, Lasso
from sklearn.ensemble import RandomForestClassifier, VotingClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, matthews_corrcoef, precision_score, recall_score, confusion_matrix
from xgboost import XGBClassifier
from joblib import Parallel, delayed
import joblib
import os
from scipy.stats import uniform, randint, loguniform
from imblearn.over_sampling import SMOTE, BorderlineSMOTE, ADASYN
from imblearn.pipeline import Pipeline
from imblearn.metrics import classification_report_imbalanced
import optuna
from optuna.integration import XGBoostPruningCallback
import gc

print("Reading CSV...")
df = pd.read_csv('cross_mapped_data_with_tanimoto_cancer.csv', low_memory=False)
print(f"Full dataset shape: {df.shape}")

# 1. Extract paired features
print("Extracting paired features...")
feature_pairs = {}
for col in df.columns:
    if "_" in col and col not in ["smiley_1", "smiley_2", "drug_1", "drug_2", "matched"]:
        base = "_".join(col.split("_")[:-1])
        feature_pairs.setdefault(base, []).append(col)

paired_features = [pair for pair in feature_pairs.values() if len(pair) == 2]
flattened_features = [f for pair in paired_features for f in pair]

# Define all Tanimoto features
tanimoto_features = [
    "Tanimoto_Morgan",
    "Tanimoto_FeatMorgan",
    "Tanimoto_AtomPair",
    "Tanimoto_RDKit",
    "Tanimoto_Torsion",
    "Tanimoto_Layered",
    "Tanimoto_MACCS"
]
flattened_features.extend(tanimoto_features)

# 2. Feature Engineering
print("Performing feature engineering...")
X = df[flattened_features].copy()
y = df["matched"].values

# Free up memory
del df
gc.collect()

# Create interaction features for all Tanimoto scores
interaction_features = {}
for i in range(len(tanimoto_features)):
    for j in range(i+1, len(tanimoto_features)):
        interaction_features[f'interaction_{tanimoto_features[i]}_{tanimoto_features[j]}'] = \
            X[tanimoto_features[i]] * X[tanimoto_features[j]]

# Create ratio features for paired features
ratio_features = {}
for pair in paired_features:
    if all(f in X.columns for f in pair):
        ratio_features[f'ratio_{pair[0]}_{pair[1]}'] = X[pair[0]] / (X[pair[1]] + 1e-10)

# Add polynomial features for all Tanimoto features (reduced degree to save memory)
poly = PolynomialFeatures(degree=2, include_bias=False)
poly_features = poly.fit_transform(X[tanimoto_features])
n_original = len(tanimoto_features)
poly_feature_names = [f'poly_{i}' for i in range(n_original, poly_features.shape[1])]
poly_features_df = pd.DataFrame(poly_features[:, n_original:], 
                              columns=poly_feature_names, index=X.index)

# Combine all features efficiently
X = pd.concat([
    X,
    pd.DataFrame(interaction_features, index=X.index),
    pd.DataFrame(ratio_features, index=X.index),
    poly_features_df
], axis=1)

# Free up memory
del interaction_features, ratio_features, poly_features, poly_features_df
gc.collect()

# Remove constant and quasi-constant features
selector = VarianceThreshold(threshold=0.01)
X = pd.DataFrame(selector.fit_transform(X), columns=X.columns[selector.get_support()])

print("Total features used:", len(X.columns))

with open('selected_features.txt', 'w') as f:
    # f.write("Total features using:", len(X.columns))
    for feat in X.columns:
        f.write(f"{feat}\n")

# 3. Scale features
print("Scaling features...")
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Free up memory
del X
gc.collect()

# 4. Train-test split
print("Splitting train and test sets...")
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)

# Free up memory
del X_scaled, y
gc.collect()

# 5. Define Optuna objective functions for each model
def objective_logistic(trial):
    class_weight_ratio = trial.suggest_float('class_weight_ratio', 1.5, 2.0)
    param = {
        'C': trial.suggest_float('C', 1e-5, 1e5, log=True),
        'penalty': trial.suggest_categorical('penalty', ['l1', 'l2', 'elasticnet']),
        'solver': 'saga',
        'l1_ratio': trial.suggest_float('l1_ratio', 0, 1) if trial.params['penalty'] == 'elasticnet' else None,
        'max_iter': 5000,
        'class_weight': {0: class_weight_ratio, 1: 1.0},
        'n_jobs': 8
    }
    
    model = Pipeline([
        ('smote', BorderlineSMOTE(random_state=42, sampling_strategy=0.75)),
        ('classifier', LogisticRegression(**param))
    ])
    
    scores = cross_val_score(model, X_train, y_train, cv=5, scoring='matthews_corrcoef', n_jobs=8)
    return scores.mean()

def objective_svm(trial):
    class_weight_ratio = trial.suggest_float('class_weight_ratio', 1.5, 2.0)
    param = {
        'C': trial.suggest_float('C', 1e-5, 1e5, log=True),
        'kernel': trial.suggest_categorical('kernel', ['rbf', 'poly', 'sigmoid']),
        'gamma': trial.suggest_float('gamma', 1e-5, 1, log=True),
        'class_weight': {0: class_weight_ratio, 1: 1.0},
        'cache_size': 2000
    }
    
    model = Pipeline([
        ('smote', BorderlineSMOTE(random_state=42, sampling_strategy=0.75)),
        ('classifier', SVC(probability=True, **param))
    ])
    
    scores = cross_val_score(model, X_train, y_train, cv=5, scoring='matthews_corrcoef', n_jobs=8)
    return scores.mean()

def objective_rf(trial):
    class_weight_ratio = trial.suggest_float('class_weight_ratio', 1.5, 2.0)
    param = {
        'n_estimators': trial.suggest_int('n_estimators', 300, 1000),
        'max_depth': trial.suggest_int('max_depth', 10, 100),
        'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
        'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10),
        'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2', None]),
        'class_weight': {0: class_weight_ratio, 1: 1.0},
        'n_jobs': 8
    }
    
    model = Pipeline([
        ('smote', BorderlineSMOTE(random_state=42, sampling_strategy=0.75)),
        ('classifier', RandomForestClassifier(**param))
    ])
    
    scores = cross_val_score(model, X_train, y_train, cv=5, scoring='matthews_corrcoef', n_jobs=8)
    return scores.mean()

def objective_xgb(trial):
    param = {
        'n_estimators': trial.suggest_int('n_estimators', 300, 1000),
        'max_depth': trial.suggest_int('max_depth', 3, 15),
        'learning_rate': trial.suggest_float('learning_rate', 1e-3, 0.3, log=True),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
        'gamma': trial.suggest_float('gamma', 0, 1),
        'reg_alpha': trial.suggest_float('reg_alpha', 0, 1),
        'reg_lambda': trial.suggest_float('reg_lambda', 0, 1),
        'scale_pos_weight': trial.suggest_float('scale_pos_weight', 1.5, 2.0),
        'n_jobs': 8,
        'eval_metric': "logloss"
        # Removed use_label_encoder parameter as it's deprecated
    }
    
    model = Pipeline([
        ('smote', BorderlineSMOTE(random_state=42, sampling_strategy=0.75)),
        ('classifier', XGBClassifier(**param))
    ])
    
    scores = cross_val_score(model, X_train, y_train, cv=5, scoring='matthews_corrcoef', n_jobs=8)
    return scores.mean()

# 6. Optimize hyperparameters using Optuna with parallel processing
print("Optimizing hyperparameters...")
models = {}
best_params = {}

# Set up parallel processing for Optuna
n_jobs = 8

# Logistic Regression
study_lr = optuna.create_study(direction='maximize')
study_lr.optimize(objective_logistic, n_trials=30, n_jobs=n_jobs)
best_params_lr = study_lr.best_params.copy()
class_weight_ratio = best_params_lr.pop('class_weight_ratio')  # Remove class_weight_ratio from params
best_params['Logistic Regression'] = best_params_lr
models['Logistic Regression'] = Pipeline([
    ('smote', BorderlineSMOTE(random_state=42, sampling_strategy=0.75)),
    ('classifier', LogisticRegression(**best_params_lr, class_weight={0: class_weight_ratio, 1: 1.0}, n_jobs=8))
])

# SVM
study_svm = optuna.create_study(direction='maximize')
study_svm.optimize(objective_svm, n_trials=30, n_jobs=n_jobs)
best_params_svm = study_svm.best_params.copy()
class_weight_ratio = best_params_svm.pop('class_weight_ratio')  # Remove class_weight_ratio from params
best_params['SVM'] = best_params_svm
models['SVM'] = Pipeline([
    ('smote', BorderlineSMOTE(random_state=42, sampling_strategy=0.75)),
    ('classifier', SVC(probability=True, class_weight={0: class_weight_ratio, 1: 1.0}, **best_params_svm))
])

# Random Forest
study_rf = optuna.create_study(direction='maximize')
study_rf.optimize(objective_rf, n_trials=30, n_jobs=n_jobs)
best_params_rf = study_rf.best_params.copy()
class_weight_ratio = best_params_rf.pop('class_weight_ratio')  # Remove class_weight_ratio from params
best_params['Random Forest'] = best_params_rf
models['Random Forest'] = Pipeline([
    ('smote', BorderlineSMOTE(random_state=42, sampling_strategy=0.75)),
    ('classifier', RandomForestClassifier(**best_params_rf, class_weight={0: class_weight_ratio, 1: 1.0}, n_jobs=8))
])

# XGBoost
study_xgb = optuna.create_study(direction='maximize')
study_xgb.optimize(objective_xgb, n_trials=30, n_jobs=n_jobs)
best_params['XGBoost'] = study_xgb.best_params
models['XGBoost'] = Pipeline([
    ('smote', BorderlineSMOTE(random_state=42, sampling_strategy=0.75)),
    ('classifier', XGBClassifier(eval_metric="logloss", **study_xgb.best_params, n_jobs=8))
])

# 7. Train and evaluate models
print("\nTraining and evaluating models...")
results = {}

os.makedirs('saved_model_cancer', exist_ok=True)  # Ensure directory exists

for model_name, model in models.items():
    print(f"\nTraining {model_name}...")
    model.fit(X_train, y_train)
    # Save the trained model immediately
    joblib.dump(model, f'saved_model_cancer/{model_name.replace(" ", "_")}.joblib')
    
    # Make predictions on test set
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else y_pred
    # Make predictions on train set
    y_pred_train = model.predict(X_train)
    y_proba_train = model.predict_proba(X_train)[:, 1] if hasattr(model, "predict_proba") else y_pred_train
    
    # Calculate test metrics
    results[model_name] = {
        'accuracy': accuracy_score(y_test, y_pred),
        'auc': roc_auc_score(y_test, y_proba),
        'f1': f1_score(y_test, y_pred),
        'mcc': matthews_corrcoef(y_test, y_pred),
        'precision': precision_score(y_test, y_pred),
        'recall': recall_score(y_test, y_pred),
        'confusion_matrix': confusion_matrix(y_test, y_pred),
        'classification_report': classification_report_imbalanced(y_test, y_pred)
    }
    # Calculate train metrics
    results[model_name + '_train'] = {
        'accuracy': accuracy_score(y_train, y_pred_train),
        'auc': roc_auc_score(y_train, y_proba_train),
        'f1': f1_score(y_train, y_pred_train),
        'mcc': matthews_corrcoef(y_train, y_pred_train),
        'precision': precision_score(y_train, y_pred_train),
        'recall': recall_score(y_train, y_pred_train),
        'confusion_matrix': confusion_matrix(y_train, y_pred_train),
        'classification_report': classification_report_imbalanced(y_train, y_pred_train)
    }
    
    print(f"\nResults for {model_name} (TEST):")
    print(f"Accuracy: {results[model_name]['accuracy']:.4f}")
    print(f"AUC: {results[model_name]['auc']:.4f}")
    print(f"F1: {results[model_name]['f1']:.4f}")
    print(f"MCC: {results[model_name]['mcc']:.4f}")
    print(f"Precision: {results[model_name]['precision']:.4f}")
    print(f"Recall: {results[model_name]['recall']:.4f}")
    print("\nConfusion Matrix:")
    print(results[model_name]['confusion_matrix'])
    print("\nClassification Report:")
    print(results[model_name]['classification_report'])
    
    print(f"\nResults for {model_name} (TRAIN):")
    print(f"Accuracy: {results[model_name + '_train']['accuracy']:.4f}")
    print(f"AUC: {results[model_name + '_train']['auc']:.4f}")
    print(f"F1: {results[model_name + '_train']['f1']:.4f}")
    print(f"MCC: {results[model_name + '_train']['mcc']:.4f}")
    print(f"Precision: {results[model_name + '_train']['precision']:.4f}")
    print(f"Recall: {results[model_name + '_train']['recall']:.4f}")
    print("\nConfusion Matrix:")
    print(results[model_name + '_train']['confusion_matrix'])
    print("\nClassification Report:")
    print(results[model_name + '_train']['classification_report'])
    print("\nBest Parameters:")
    print(best_params[model_name])

# 8. Create and evaluate ensemble
print("\nCreating ensemble model...")
ensemble = VotingClassifier(
    estimators=[(name, model) for name, model in models.items()],
    voting='soft',
    weights=[1]*len(models)
)

ensemble.fit(X_train, y_train)
y_pred_ensemble = ensemble.predict(X_test)
y_proba_ensemble = ensemble.predict_proba(X_test)[:, 1]

# Calculate ensemble metrics
print("\nEnsemble Model Results:")
print(f"Accuracy: {accuracy_score(y_test, y_pred_ensemble):.4f}")
print(f"AUC: {roc_auc_score(y_test, y_proba_ensemble):.4f}")
print(f"F1: {f1_score(y_test, y_pred_ensemble):.4f}")
print(f"MCC: {matthews_corrcoef(y_test, y_pred_ensemble):.4f}")
print(f"Precision: {precision_score(y_test, y_pred_ensemble):.4f}")
print(f"Recall: {recall_score(y_test, y_pred_ensemble):.4f}")
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred_ensemble))
print("\nClassification Report:")
print(classification_report_imbalanced(y_test, y_pred_ensemble))

# 9. Save models
print("\nSaving models...")
os.makedirs('saved_model_cancer', exist_ok=True)

for model_name, model in models.items():
    joblib.dump(model, f'saved_model_cancer/{model_name.replace(" ", "_")}.joblib')
    joblib.dump(best_params[model_name], f'saved_model_cancer/{model_name.replace(" ", "_")}_params.joblib')

joblib.dump(ensemble, 'saved_model_cancer/ensemble_model.joblib')
print("All models saved successfully.")
