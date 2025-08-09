import pandas as pd
from rdkit import Chem
from rdkit.Chem import DataStructs
from rdkit.Chem.rdFingerprintGenerator import (
    GetMorganGenerator, GetRDKitFPGenerator, GetTopologicalTorsionGenerator, GetAtomPairGenerator
)
from rdkit.Chem import rdMolDescriptors
from rdkit.Chem.rdmolops import RDKFingerprint
from rdkit.Chem import rdFMCS
from tqdm import tqdm

# Load dataset
df = pd.read_csv("cancer_final_data.csv")

# Ensure missing values are handled
df = df.dropna(subset=["smiley_1", "smiley_2"])  # Drop rows with missing SMILES

# Fingerprint cache
fingerprint_cache = {}

def get_fingerprint(smiles, fp_type):
    """Generate molecular fingerprints based on type with caching."""
    if (smiles, fp_type) in fingerprint_cache:
        return fingerprint_cache[(smiles, fp_type)]
    
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    if fp_type == 'Morgan':
        fp = GetMorganGenerator(radius=2, fpSize=2048).GetFingerprint(mol)
    elif fp_type == 'FeatMorgan':
        fp = GetMorganGenerator(radius=2, fpSize=2048, includeChirality=True).GetFingerprint(mol)
    elif fp_type == 'AtomPair':
        fp = GetAtomPairGenerator(fpSize=2048).GetFingerprint(mol)
    elif fp_type == 'RDKit':
        fp = GetRDKitFPGenerator(fpSize=2048).GetFingerprint(mol)
    elif fp_type == 'Torsion':
        fp = GetTopologicalTorsionGenerator(fpSize=2048).GetFingerprint(mol)
    elif fp_type == 'Layered':
        fp = RDKFingerprint(mol)
    elif fp_type == 'MACCS':
        fp = rdMolDescriptors.GetMACCSKeysFingerprint(mol)
    else:
        raise ValueError(f"Unknown fingerprint type: {fp_type}")
    
    fingerprint_cache[(smiles, fp_type)] = fp
    return fp

def tanimoto_similarity(smiles1, smiles2, fp_type):
    """Compute Tanimoto similarity between two molecules."""
    fp1 = get_fingerprint(smiles1, fp_type)
    fp2 = get_fingerprint(smiles2, fp_type)

    if fp1 is None or fp2 is None:
        return None
    return DataStructs.FingerprintSimilarity(fp1, fp2)

def compute_tanimoto(fp_type):
    """Sequential computation of Tanimoto similarity."""
    results = []
    for _, row in tqdm(df.iterrows(), total=len(df), desc=f"Computing Tanimoto {fp_type}"):
        results.append(tanimoto_similarity(row["smiley_1"], row["smiley_2"], fp_type))
    return results

def compute_mcs(smiles1, smiles2):
    """Compute Maximum Common Substructure (MCS) similarity features."""
    mol1, mol2 = Chem.MolFromSmiles(smiles1), Chem.MolFromSmiles(smiles2)
    if mol1 is None or mol2 is None:
        return None, None, None

    mcs_result = rdFMCS.FindMCS([mol1, mol2])
    mcs_size = mcs_result.numAtoms
    mcs_tanimoto = mcs_size / min(mol1.GetNumAtoms(), mol2.GetNumAtoms())
    mcs_overlap = mcs_size / max(mol1.GetNumAtoms(), mol2.GetNumAtoms())

    return mcs_size, mcs_tanimoto, mcs_overlap

def compute_mcs_features():
    """Sequential computation of MCS features."""
    mcs_sizes, mcs_tanimotos, mcs_overlaps = [], [], []
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Computing MCS Features"):
        mcs_size, mcs_tanimoto, mcs_overlap = compute_mcs(row["smiley_1"], row["smiley_2"])
        mcs_sizes.append(mcs_size)
        mcs_tanimotos.append(mcs_tanimoto)
        mcs_overlaps.append(mcs_overlap)
    return mcs_sizes, mcs_tanimotos, mcs_overlaps

# Compute Tanimoto similarities for all fingerprint types
fp_types = ['Morgan', 'FeatMorgan', 'AtomPair', 'RDKit', 'Torsion', 'Layered', 'MACCS']
for fp in fp_types:
    df[f"Tanimoto_{fp}"] = compute_tanimoto(fp)

# Compute MCS Features
# df["MCS_Size"], df["MCS_Tanimoto"], df["MCS_Overlap"] = compute_mcs_features()

# Save updated dataframe
df.to_csv("cross_mapped_data_with_tanimoto_cancer.csv", index=False)
print("Feature extraction complete!")

# import pandas as pd
# import numpy as np
# from sklearn.model_selection import train_test_split
# from sklearn.feature_selection import mutual_info_classif, RFE, SelectFromModel, SelectKBest, f_classif
# from sklearn.linear_model import LogisticRegression, Lasso
# from sklearn.ensemble import RandomForestClassifier
# from xgboost import XGBClassifier
# from sklearn.svm import SVC
# from sklearn.metrics import accuracy_score, roc_auc_score

# # Load dataset
# df = pd.read_csv("cross_mapped_data_with_tanimoto_paper.csv")  # Updated filename

# # Extract feature pairs
# feature_pairs = {}
# for col in df.columns:
#     if "_" in col and col not in ["smiley_1", "smiley_2", "drug_1", "drug_2", "matched"]:
#         base = "_".join(col.split("_")[:-1])  # Generalized base feature extraction
#         if base not in feature_pairs:
#             feature_pairs[base] = []
#         feature_pairs[base].append(col)

# # Ensure pairs are correctly grouped
# paired_features = [pair for pair in feature_pairs.values() if len(pair) == 2]
# flattened_features = [feat for pair in paired_features for feat in pair]

# # Add new Tanimoto and MCS features
# additional_features = [
#     "Tanimoto_Morgan", "Tanimoto_FeatMorgan", "Tanimoto_AtomPair",
#     "Tanimoto_RDKit", "Tanimoto_Torsion", "Tanimoto_Layered", "Tanimoto_MACCS",
#     "MCS_Size", "MCS_Tanimoto", "MCS_Overlap"
# ]
# flattened_features.extend(additional_features)

# # Define X and y
# X = df[flattened_features]
# y = df["matched"]

# # Train-test split
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ### Feature Selection Methods
# # 1. Mutual Information
# mi_scores = mutual_info_classif(X_train, y_train)
# mi_scores_dict = {feat: mi for feat, mi in zip(flattened_features, mi_scores)}
# selected_mi = sorted(mi_scores_dict, key=mi_scores_dict.get, reverse=True)[:20]

# # 2. RFE with Logistic Regression
# log_reg = LogisticRegression(max_iter=1000)
# rfe = RFE(log_reg, n_features_to_select=20)
# rfe.fit(X_train, y_train)
# selected_rfe = [feat for feat, keep in zip(flattened_features, rfe.support_) if keep]

# # 3. Random Forest Feature Importance
# rf = RandomForestClassifier(n_estimators=100, random_state=42)
# rf.fit(X_train, y_train)
# rf_importances = {feat: imp for feat, imp in zip(flattened_features, rf.feature_importances_)}
# selected_rf = sorted(rf_importances, key=rf_importances.get, reverse=True)[:20]

# # 4. Lasso Regression (L1 Regularization)
# lasso = Lasso(alpha=0.01)
# lasso.fit(X_train, y_train)
# lasso_coeffs = {feat: coef for feat, coef in zip(flattened_features, lasso.coef_)}
# selected_lasso = [feat for feat, coef in lasso_coeffs.items() if abs(coef) > 0]

# # 5. XGBoost Feature Importance
# xgb = XGBClassifier(use_label_encoder=False, eval_metric="logloss")
# xgb.fit(X_train, y_train)
# xgb_importances = {feat: imp for feat, imp in zip(flattened_features, xgb.feature_importances_)}
# selected_xgb = sorted(xgb_importances, key=xgb_importances.get, reverse=True)[:20]

# # 6. SelectKBest (ANOVA F-score)
# kbest = SelectKBest(score_func=f_classif, k=20)
# kbest.fit(X_train, y_train)
# selected_kbest = [feat for feat, keep in zip(flattened_features, kbest.get_support()) if keep]

# ### Ensure features are kept in pairs
# def enforce_feature_pairs(selected_features):
#     selected_pairs = []
#     for pair in paired_features:
#         if any(f in selected_features for f in pair):
#             selected_pairs.extend(pair)
#     return list(set(selected_pairs + [f for f in selected_features if f in additional_features]))

# selected_mi_pairs = enforce_feature_pairs(selected_mi)
# selected_rfe_pairs = enforce_feature_pairs(selected_rfe)
# selected_rf_pairs = enforce_feature_pairs(selected_rf)
# selected_lasso_pairs = enforce_feature_pairs(selected_lasso)
# selected_xgb_pairs = enforce_feature_pairs(selected_xgb)
# selected_kbest_pairs = enforce_feature_pairs(selected_kbest)

# ### Train Models with Selected Features
# def train_model(X_train, X_test, y_train, y_test, features, model):
#     X_train_selected = X_train[features]
#     X_test_selected = X_test[features]
#     model.fit(X_train_selected, y_train)
#     y_pred = model.predict(X_test_selected)
#     y_pred_proba = model.predict_proba(X_test_selected)[:, 1] if hasattr(model, "predict_proba") else y_pred
#     return accuracy_score(y_test, y_pred), roc_auc_score(y_test, y_pred_proba)

# models = {
#     "Logistic Regression": LogisticRegression(max_iter=1000),
#     "SVM": SVC(probability=True),  # Enable probability for AUC calculation
#     "Random Forest": RandomForestClassifier(n_estimators=100),
#     "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric="logloss"),
# }

# accuracy_results = {}
# auc_results = {}

# for name, model in models.items():
#     accuracy_results[name] = {}
#     auc_results[name] = {}
#     for method, selected_features in {
#         "Mutual Information": selected_mi_pairs,
#         "RFE": selected_rfe_pairs,
#         "Random Forest": selected_rf_pairs,
#         "Lasso": selected_lasso_pairs,
#         "XGBoost": selected_xgb_pairs,
#         "SelectKBest": selected_kbest_pairs,
#     }.items():
#         acc, auc = train_model(X_train, X_test, y_train, y_test, selected_features, model)
#         accuracy_results[name][method] = acc
#         auc_results[name][method] = auc

# ### Print Results
# print("\nFeature Selection Results (Top Selected Features in Pairs):")
# print("Mutual Information:", selected_mi_pairs)
# print("RFE:", selected_rfe_pairs)
# print("Random Forest:", selected_rf_pairs)
# print("Lasso:", selected_lasso_pairs)
# print("XGBoost:", selected_xgb_pairs)
# print("SelectKBest:", selected_kbest_pairs)

# print("\nModel Performance with Selected Features:")
# for model in models.keys():
#     print(f"\n{model}:")
#     for method in accuracy_results[model].keys():
#         print(f"  {method}: Accuracy = {accuracy_results[model][method]:.4f}, AUC = {auc_results[model][method]:.4f}")
