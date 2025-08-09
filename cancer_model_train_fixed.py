import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.feature_selection import mutual_info_classif, SelectKBest, f_classif, VarianceThreshold
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, matthews_corrcoef, precision_score, recall_score, confusion_matrix
from xgboost import XGBClassifier
import joblib
import os
from imblearn.over_sampling import BorderlineSMOTE
from imblearn.pipeline import Pipeline
from imblearn.metrics import classification_report_imbalanced
import optuna
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

# 2. Feature Engineering (Reduced complexity)
print("Performing feature engineering...")
X = df[flattened_features].copy()
y = df["matched"].values

# Free up memory
del df
gc.collect()

# Create interaction features for Tanimoto scores (reduced)
interaction_features = {}
for i in range(len(tanimoto_features)):
    for j in range(i+1, min(i+3, len(tanimoto_features))):  # Limit interactions
        interaction_features[f'interaction_{tanimoto_features[i]}_{tanimoto_features[j]}'] = \
            X[tanimoto_features[i]] * X[tanimoto_features[j]]

# Create ratio features for paired features (reduced)
ratio_features = {}
for pair in paired_features[:50]:  # Limit to first 50 pairs
    if all(f in X.columns for f in pair):
        ratio_features[f'ratio_{pair[0]}_{pair[1]}'] = X[pair[0]] / (X[pair[1]] + 1e-10)

# Add polynomial features for Tanimoto features (degree 1 only - linear combinations)
poly = PolynomialFeatures(degree=1, include_bias=False)
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

# Feature selection to reduce dimensionality
print("Performing feature selection...")
# Select top 1000 features based on mutual information
selector_mi = SelectKBest(score_func=mutual_info_classif, k=min(1000, X.shape[1]))
X_selected = selector_mi.fit_transform(X, y)
selected_features = X.columns[selector_mi.get_support()]
X = pd.DataFrame(X_selected, columns=selected_features, index=X.index)

print("Total features used:", len(X.columns))

with open('selected_features_fixed.txt', 'w') as f:
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

# Check class distribution
print(f"Training set class distribution: {np.bincount(y_train)}")
print(f"Test set class distribution: {np.bincount(y_test)}")

# Calculate safe SMOTE ratio - more conservative approach
# minority_count = min(np.bincount(y_train))
# majority_count = max(np.bincount(y_train))
# # Use a more conservative ratio that ensures we only add samples, never remove
# safe_ratio = min(0.6, (minority_count + 100) / majority_count)  # Add buffer and cap at 0.6
# print(f"Safe SMOTE ratio: {safe_ratio:.3f}")

# Alternative: Use 'auto' strategy which is more robust
print("Using 'auto' SMOTE strategy for better stability")

# Free up memory
del X_scaled, y
gc.collect()

# 5. Define Optuna objective functions for each model (with regularization)
def objective_logistic(trial):
    class_weight_ratio = trial.suggest_float('class_weight_ratio', 1.2, 1.8)
    penalty = trial.suggest_categorical('penalty', ['l1', 'l2'])  # Removed elasticnet
    
    # Ensure solver is compatible with penalty
    if penalty == 'l1':
        solver = 'liblinear'  # liblinear supports l1
    else:  # l2
        solver = 'liblinear'  # liblinear also supports l2
    
    param = {
        'C': trial.suggest_float('C', 0.1, 10.0, log=True),  # Reduced range
        'penalty': penalty,
        'solver': solver,
        'max_iter': 1000,  # Reduced iterations
        'class_weight': {0: class_weight_ratio, 1: 1.0}
        # Removed n_jobs for liblinear solver
    }
    
    try:
        model = Pipeline([
            ('smote', BorderlineSMOTE(random_state=42, sampling_strategy='auto')),  # Use auto strategy
            ('classifier', LogisticRegression(**param))
        ])
        
        # Use stratified k-fold with fewer folds
        cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
        scores = cross_val_score(model, X_train, y_train, cv=cv, scoring='matthews_corrcoef', n_jobs=4)
        return scores.mean()
    except Exception as e:
        print(f"Error in objective_logistic: {e}")
        print(f"Parameters: {param}")
        # Return a very low score to indicate this combination is invalid
        return -1.0

def objective_svm(trial):
    class_weight_ratio = trial.suggest_float('class_weight_ratio', 1.2, 1.8)
    param = {
        'C': trial.suggest_float('C', 0.1, 10.0, log=True),  # Reduced range
        'kernel': trial.suggest_categorical('kernel', ['rbf', 'linear']),  # Removed poly and sigmoid
        'gamma': trial.suggest_float('gamma', 1e-4, 1e-1, log=True),  # Reduced range
        'class_weight': {0: class_weight_ratio, 1: 1.0},
        'cache_size': 1000  # Reduced cache size
    }
    
    model = Pipeline([
        ('smote', BorderlineSMOTE(random_state=42, sampling_strategy='auto')),  # Use auto strategy
        ('classifier', SVC(probability=True, **param))
    ])
    
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    scores = cross_val_score(model, X_train, y_train, cv=cv, scoring='matthews_corrcoef', n_jobs=4)
    return scores.mean()

def objective_rf(trial):
    class_weight_ratio = trial.suggest_float('class_weight_ratio', 1.2, 1.8)
    param = {
        'n_estimators': trial.suggest_int('n_estimators', 100, 300),  # Reduced range
        'max_depth': trial.suggest_int('max_depth', 5, 20),  # Reduced depth
        'min_samples_split': trial.suggest_int('min_samples_split', 5, 20),
        'min_samples_leaf': trial.suggest_int('min_samples_leaf', 2, 10),
        'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2']),  # Removed None
        'class_weight': {0: class_weight_ratio, 1: 1.0},
        'n_jobs': 4
    }
    
    model = Pipeline([
        ('smote', BorderlineSMOTE(random_state=42, sampling_strategy='auto')),  # Use auto strategy
        ('classifier', RandomForestClassifier(**param))
    ])
    
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    scores = cross_val_score(model, X_train, y_train, cv=cv, scoring='matthews_corrcoef', n_jobs=4)
    return scores.mean()

def objective_xgb(trial):
    param = {
        'n_estimators': trial.suggest_int('n_estimators', 100, 300),  # Reduced range
        'max_depth': trial.suggest_int('max_depth', 3, 8),  # Reduced depth
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2, log=True),  # Reduced range
        'subsample': trial.suggest_float('subsample', 0.7, 0.9),  # Reduced range
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.7, 0.9),  # Reduced range
        'min_child_weight': trial.suggest_int('min_child_weight', 3, 10),
        'gamma': trial.suggest_float('gamma', 0.1, 1.0),  # Increased minimum
        'reg_alpha': trial.suggest_float('reg_alpha', 0.1, 1.0),  # Increased minimum
        'reg_lambda': trial.suggest_float('reg_lambda', 0.1, 1.0),  # Increased minimum
        'scale_pos_weight': trial.suggest_float('scale_pos_weight', 1.2, 1.8),
        'n_jobs': 4,
        'eval_metric': "logloss"
    }
    
    model = Pipeline([
        ('smote', BorderlineSMOTE(random_state=42, sampling_strategy='auto')),  # Use auto strategy
        ('classifier', XGBClassifier(**param))
    ])
    
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    scores = cross_val_score(model, X_train, y_train, cv=cv, scoring='matthews_corrcoef', n_jobs=4)
    return scores.mean()

# 6. Optimize hyperparameters using Optuna with reduced trials
print("Optimizing hyperparameters...")
models = {}
best_params = {}

# Set up parallel processing for Optuna
n_jobs = 4  # Reduced parallel jobs

# Logistic Regression
study_lr = optuna.create_study(direction='maximize')
study_lr.optimize(objective_logistic, n_trials=15, n_jobs=n_jobs)  # Reduced trials
best_params_lr = study_lr.best_params.copy()
class_weight_ratio = best_params_lr.pop('class_weight_ratio')
# Add solver and max_iter parameters that were hardcoded in objective function
best_params_lr['solver'] = 'liblinear'
best_params_lr['max_iter'] = 1000
best_params['Logistic Regression'] = best_params_lr

# Debug: Print the parameters being passed to LogisticRegression
print(f"Logistic Regression parameters: {best_params_lr}")
print(f"Class weight ratio: {class_weight_ratio}")

# Create LogisticRegression with explicit parameters to ensure solver is set correctly
# Ensure penalty is compatible with liblinear solver
penalty = best_params_lr['penalty']
if penalty == 'l1':
    solver = 'liblinear'  # liblinear supports l1
elif penalty == 'l2':
    solver = 'liblinear'  # liblinear also supports l2
else:
    penalty = 'l2'  # Default to l2 if something else
    solver = 'liblinear'

print(f"Final penalty: {penalty}, solver: {solver}")

# Create a clean parameter dictionary for LogisticRegression
lr_params = {
    'C': best_params_lr['C'],
    'penalty': penalty,
    'solver': solver,
    'max_iter': 1000,
    'class_weight': {0: class_weight_ratio, 1: 1.0}
}

print(f"Final LR parameters: {lr_params}")

# Test the LogisticRegression creation to catch any errors early
try:
    lr_classifier = LogisticRegression(**lr_params)
    print("LogisticRegression created successfully")
except Exception as e:
    print(f"Error creating LogisticRegression: {e}")
    # Fallback to safe parameters
    lr_params = {
        'C': 1.0,
        'penalty': 'l2',
        'solver': 'liblinear',
        'max_iter': 1000,
        'class_weight': {0: class_weight_ratio, 1: 1.0}
    }
    lr_classifier = LogisticRegression(**lr_params)
    print("Using fallback parameters for LogisticRegression")

models['Logistic Regression'] = Pipeline([
    ('smote', BorderlineSMOTE(random_state=42, sampling_strategy='auto')),
    ('classifier', lr_classifier)
])

# SVM
study_svm = optuna.create_study(direction='maximize')
study_svm.optimize(objective_svm, n_trials=15, n_jobs=n_jobs)
best_params_svm = study_svm.best_params.copy()
class_weight_ratio = best_params_svm.pop('class_weight_ratio')
best_params['SVM'] = best_params_svm
models['SVM'] = Pipeline([
    ('smote', BorderlineSMOTE(random_state=42, sampling_strategy='auto')),
    ('classifier', SVC(probability=True, class_weight={0: class_weight_ratio, 1: 1.0}, **best_params_svm))
])

# Random Forest
study_rf = optuna.create_study(direction='maximize')
study_rf.optimize(objective_rf, n_trials=15, n_jobs=n_jobs)
best_params_rf = study_rf.best_params.copy()
class_weight_ratio = best_params_rf.pop('class_weight_ratio')
best_params['Random Forest'] = best_params_rf
models['Random Forest'] = Pipeline([
    ('smote', BorderlineSMOTE(random_state=42, sampling_strategy='auto')),
    ('classifier', RandomForestClassifier(**best_params_rf, class_weight={0: class_weight_ratio, 1: 1.0}, n_jobs=4))
])

# XGBoost
study_xgb = optuna.create_study(direction='maximize')
study_xgb.optimize(objective_xgb, n_trials=15, n_jobs=n_jobs)
best_params['XGBoost'] = study_xgb.best_params
models['XGBoost'] = Pipeline([
    ('smote', BorderlineSMOTE(random_state=42, sampling_strategy='auto')),
    ('classifier', XGBClassifier(eval_metric="logloss", **study_xgb.best_params, n_jobs=4))
])

# 7. Train and evaluate models
print("\nTraining and evaluating models...")
results = {}

os.makedirs('saved_model_cancer_fixed', exist_ok=True)

for model_name, model in models.items():
    print(f"\nTraining {model_name}...")
    try:
        model.fit(X_train, y_train)
        print(f"{model_name} trained successfully")
        
        # Save the trained model immediately
        joblib.dump(model, f'saved_model_cancer_fixed/{model_name.replace(" ", "_")}.joblib')
    except Exception as e:
        print(f"Error training {model_name}: {e}")
        print(f"Model parameters: {model.get_params()}")
        continue
    
    # Make predictions on test set
    try:
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else y_pred
        
        # Make predictions on train set
        y_pred_train = model.predict(X_train)
        y_proba_train = model.predict_proba(X_train)[:, 1] if hasattr(model, "predict_proba") else y_pred_train
    except Exception as e:
        print(f"Error making predictions for {model_name}: {e}")
        continue
    
    # Calculate test metrics
    try:
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
    except Exception as e:
        print(f"Error calculating metrics for {model_name}: {e}")
        continue
    
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
    
    # Calculate overfitting metric
    train_acc = results[model_name + '_train']['accuracy']
    test_acc = results[model_name]['accuracy']
    overfitting_gap = train_acc - test_acc
    
    print(f"\nResults for {model_name} (TEST):")
    print(f"Accuracy: {results[model_name]['accuracy']:.4f}")
    print(f"AUC: {results[model_name]['auc']:.4f}")
    print(f"F1: {results[model_name]['f1']:.4f}")
    print(f"MCC: {results[model_name]['mcc']:.4f}")
    print(f"Precision: {results[model_name]['precision']:.4f}")
    print(f"Recall: {results[model_name]['recall']:.4f}")
    print(f"Overfitting Gap (Train-Test): {overfitting_gap:.4f}")
    
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

# Calculate ensemble overfitting
y_pred_ensemble_train = ensemble.predict(X_train)
ensemble_train_acc = accuracy_score(y_train, y_pred_ensemble_train)
ensemble_test_acc = accuracy_score(y_test, y_pred_ensemble)
ensemble_overfitting_gap = ensemble_train_acc - ensemble_test_acc
print(f"Ensemble Overfitting Gap (Train-Test): {ensemble_overfitting_gap:.4f}")

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred_ensemble))
print("\nClassification Report:")
print(classification_report_imbalanced(y_test, y_pred_ensemble))

# 9. Save models and results
print("\nSaving models...")
os.makedirs('saved_model_cancer_fixed', exist_ok=True)

for model_name, model in models.items():
    joblib.dump(model, f'saved_model_cancer_fixed/{model_name.replace(" ", "_")}.joblib')
    joblib.dump(best_params[model_name], f'saved_model_cancer_fixed/{model_name.replace(" ", "_")}_params.joblib')

joblib.dump(ensemble, 'saved_model_cancer_fixed/ensemble_model.joblib')

# Save results summary
results_summary = {}
for model_name in models.keys():
    results_summary[model_name] = {
        'test_accuracy': results[model_name]['accuracy'],
        'test_auc': results[model_name]['auc'],
        'test_f1': results[model_name]['f1'],
        'test_mcc': results[model_name]['mcc'],
        'train_accuracy': results[model_name + '_train']['accuracy'],
        'overfitting_gap': results[model_name + '_train']['accuracy'] - results[model_name]['accuracy']
    }

# Add ensemble results
results_summary['Ensemble'] = {
    'test_accuracy': accuracy_score(y_test, y_pred_ensemble),
    'test_auc': roc_auc_score(y_test, y_proba_ensemble),
    'test_f1': f1_score(y_test, y_pred_ensemble),
    'test_mcc': matthews_corrcoef(y_test, y_pred_ensemble),
    'train_accuracy': ensemble_train_acc,
    'overfitting_gap': ensemble_overfitting_gap
}

joblib.dump(results_summary, 'saved_model_cancer_fixed/results_summary.joblib')

print("All models saved successfully.")
print("\n=== RESULTS SUMMARY ===")
for model_name, metrics in results_summary.items():
    print(f"\n{model_name}:")
    print(f"  Test Accuracy: {metrics['test_accuracy']:.4f}")
    print(f"  Test AUC: {metrics['test_auc']:.4f}")
    print(f"  Test F1: {metrics['test_f1']:.4f}")
    print(f"  Test MCC: {metrics['test_mcc']:.4f}")
    print(f"  Train Accuracy: {metrics['train_accuracy']:.4f}")
    print(f"  Overfitting Gap: {metrics['overfitting_gap']:.4f}") 