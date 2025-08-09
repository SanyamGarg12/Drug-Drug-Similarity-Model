import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, roc_auc_score
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import rdMolDescriptors
import networkx as nx
from rdkit.Chem import rdmolops

def get_graph_features(smiles):
    """
    Extract graph-based features from a molecule using RDKit and NetworkX.
    
    Args:
        smiles (str): SMILES string of the molecule
        
    Returns:
        np.array: Array of graph-based features
    """
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            print(f"Warning: Invalid SMILES string: {smiles}")
            return np.zeros(10)  # Return zeros if molecule is invalid
        
        # Get molecular graph
        G = nx.Graph()
        for atom in mol.GetAtoms():
            G.add_node(atom.GetIdx(), atomic_num=atom.GetAtomicNum())
        for bond in mol.GetBonds():
            G.add_edge(bond.GetBeginAtomIdx(), bond.GetEndAtomIdx(), bond_type=bond.GetBondType())
        
        # Calculate graph features
        features = []
        # 1. Number of nodes (atoms)
        features.append(G.number_of_nodes())
        # 2. Number of edges (bonds)
        features.append(G.number_of_edges())
        # 3. Average node degree
        features.append(sum(dict(G.degree()).values()) / G.number_of_nodes() if G.number_of_nodes() > 0 else 0)
        # 4. Graph density
        features.append(nx.density(G))
        # 5. Number of connected components
        features.append(nx.number_connected_components(G))
        # 6. Average clustering coefficient
        features.append(nx.average_clustering(G))
        # 7. Graph diameter
        features.append(nx.diameter(G) if nx.is_connected(G) else 0)
        # 8. Average shortest path length
        features.append(nx.average_shortest_path_length(G) if nx.is_connected(G) else 0)
        # 9. Number of rings
        features.append(rdMolDescriptors.CalcNumRings(mol))
        # 10. Number of aromatic rings
        features.append(rdMolDescriptors.CalcNumAromaticRings(mol))
        
        return np.array(features)
    except Exception as e:
        print(f"Error processing SMILES {smiles}: {str(e)}")
        return np.zeros(10)

def prepare_features(df):
    """
    Prepare features for drug pairs by combining existing features with graph features.
    
    Args:
        df (pd.DataFrame): DataFrame containing drug pairs and their features
        
    Returns:
        pd.DataFrame: DataFrame with combined features
    """
    # Validate input data
    required_columns = ['smiley_1', 'smiley_2', 'matched']
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")
    
    print("Extracting graph features...")
    print(f"Original dataframe shape: {df.shape}")
    
    # Extract graph features for both drugs
    graph_features_1 = np.array([get_graph_features(smiles) for smiles in tqdm(df['smiley_1'])])
    graph_features_2 = np.array([get_graph_features(smiles) for smiles in tqdm(df['smiley_2'])])
    
    print(f"Graph features 1 shape: {graph_features_1.shape}")
    print(f"Graph features 2 shape: {graph_features_2.shape}")
    
    # Calculate differences between graph features
    graph_feature_diffs = np.abs(graph_features_1 - graph_features_2)
    
    # Create feature names for graph features
    graph_feature_names = [
        'graph_num_atoms_diff', 'graph_num_bonds_diff', 'graph_avg_degree_diff', 
        'graph_density_diff', 'graph_connected_components_diff', 
        'graph_clustering_coeff_diff', 'graph_diameter_diff',
        'graph_avg_path_length_diff', 'graph_num_rings_diff', 
        'graph_num_aromatic_rings_diff'
    ]
    
    # Create DataFrame for graph features
    X_graph = pd.DataFrame(graph_feature_diffs, columns=graph_feature_names, index=df.index)
    
    # Select existing features from the dataset
    # Exclude columns that are not features (like drug names, SMILES, etc.)
    feature_columns = [col for col in df.columns if col not in [
        'drug_1', 'drug_2', 'matched', 'common_targets', 'smiley_1', 'smiley_2'
    ]]
    
    print(f"Number of feature columns: {len(feature_columns)}")
    print(f"Existing features shape: {df[feature_columns].shape}")
    print(f"Graph features shape: {X_graph.shape}")
    
    # Validate shapes before concatenation
    if len(df) != len(X_graph):
        raise ValueError(f"Shape mismatch: df has {len(df)} rows but X_graph has {len(X_graph)} rows")
    
    # Combine existing features with graph features
    X_combined = pd.concat([df[feature_columns], X_graph], axis=1)
    print(f"Combined features shape: {X_combined.shape}")
    
    # Reset index to ensure consistent indexing
    X_combined = X_combined.reset_index(drop=True)
    
    return X_combined

def train_model(df, test_size=0.2, random_state=42):
    """
    Train models using combined features.
    
    Args:
        df (pd.DataFrame): DataFrame containing drug pairs and their features
        test_size (float): Proportion of data to use for testing
        random_state (int): Random seed for reproducibility
        
    Returns:
        dict: Dictionary containing model performance metrics
    """
    # Validate input parameters
    if not 0 < test_size < 1:
        raise ValueError("test_size must be between 0 and 1")
    
    # Prepare features
    X = prepare_features(df)
    y = df["matched"].values
    
    print(f"\nFinal shapes before train_test_split:")
    print(f"X shape: {X.shape}")
    print(f"y shape: {y.shape}")
    
    # Validate shapes before splitting
    if len(X) != len(y):
        raise ValueError(f"Shape mismatch: X has {len(X)} samples but y has {len(y)} samples")
    
    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    # Define models
    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "SVM": SVC(probability=True),
        "Random Forest": RandomForestClassifier(n_estimators=100),
        "XGBoost": XGBClassifier(eval_metric="logloss"),
    }
    
    # Train and evaluate models
    results = {}
    for model_name, model in tqdm(models.items()):
        try:
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            y_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else y_pred
            
            results[model_name] = {
                "accuracy": accuracy_score(y_test, y_pred),
                "auc": roc_auc_score(y_test, y_proba)
            }
            
            # Print feature importance for tree-based models
            if hasattr(model, "feature_importances_"):
                print(f"\nFeature importance for {model_name}:")
                feature_importance = pd.DataFrame({
                    'feature': X.columns,
                    'importance': model.feature_importances_
                }).sort_values('importance', ascending=False)
                print(feature_importance.head(20))  # Show top 20 most important features
        except Exception as e:
            print(f"Error training {model_name}: {str(e)}")
            results[model_name] = {"error": str(e)}
    
    return results

if __name__ == "__main__":
    try:
        # Load data
        print("Loading data...")
        df = pd.read_csv('training_data_paper.csv', low_memory=False)
        print(f"Loaded data shape: {df.shape}")
        
        # Train models on full dataset
        print("\nTraining models...")
        results = train_model(df)
        
        # Print results
        print("\nModel Performance:")
        for model_name, metrics in results.items():
            print(f"\n{model_name}:")
            if "error" in metrics:
                print(f"  Error: {metrics['error']}")
            else:
                print(f"  Accuracy: {metrics['accuracy']:.4f}")
                print(f"  AUC: {metrics['auc']:.4f}")
    except Exception as e:
        print(f"Error in main execution: {str(e)}") 