import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from joblib import dump, load
import os
import sys
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler

# Chemin du fichier CSV
csv_file_path = "nomo32_metadata.csv"

# Liste des noms des coordonnées
coordinate_names = [
    "HIP_Circ", "Seat_Back_Angle", "Calf_Left", "Calf_Right", "Across_Back", "NeckBase_Circ", "CHEST_Circ", "Neck_Circ",
    "TrouserWaist_Height_Back", "TrouserWAIST_Circ", "MaxWAIST_Circ", "Shoulder_to_Shoulder", "Across_Front",
    "CrotchLength_Front", "CrotchLength_Back", "Thigh_Circ", "Knee_Circ", "Calf_Circ", "Outseam", "Inseam", "Knee_Height",
    "Shoulder_Length", "Shoulder_to_Wrist", "Bicep_Circ", "Elbow_Circ", "Wrist_Circ", "Shoulder_Slope_Left",
    "Shoulder_Slope_Right", "Joints", "Landmarks", "CROTCH_Height", "Head_Top_Height", "NECK_Height", "MaxWaist_Height",
    "TrouserWaist_Height_Front", "Ankle_Girth_Left", "Ankle_Girth_Right", "Calf_Height", "Ankle_Height", "Ankle_Circ",
    "Thigh_Height", "Shoulder_Width_thruTheBody", "Hip_Width_ThruTheBody", "Overarm_Width_ThruTheBody", "Waist_Height_Back_EZ",
    "Hip_Height", "Bust_to_Bust", "Underbust_Circ", "BUST_Circ", "SideNeck_to_Bust", "NeckSide_to_Wrist",
    "Shoulder_to_floor_Right", "Shoulder_to_floor_Left", "TopHip_Circ", "Bust_Height", "Torso_Height", "Istumakorkeus",
    "NaturalWAIST_Circ", "NaturalWaist_Height", "Shoulder_Width_ThruTheBody", "Hip_2_Width_ThruTheBody", "Hip_2_Circ", "Hip_2_Height"
]

# Fonction pour charger et préparer les données
def load_and_prepare_data(csv_file_path):
    df = pd.read_csv(csv_file_path)
    # Séparer les données par genre avant de supprimer la colonne 'photo_names'
    male_mask = df['photo_names'].str.contains(r'^male\d{4}$', na=False)
    female_mask = df['photo_names'].str.contains(r'^female\d{4}$', na=False)
    df_male = df[male_mask].copy()
    df_female = df[female_mask].copy()
    df_male.drop(columns=['photo_names'], inplace=True)
    df_female.drop(columns=['photo_names'], inplace=True)
    numeric_columns = df.select_dtypes(include=['number']).columns
    df_numeric_m = df_male[numeric_columns].dropna(axis=1, how='all')
    df_numeric_f = df_female[numeric_columns].dropna(axis=1, how='all')
    imputer = SimpleImputer(strategy='mean')
    df_imputed_m = pd.DataFrame(imputer.fit_transform(df_numeric_m), columns=df_numeric_m.columns)
    df_imputed_f = pd.DataFrame(imputer.fit_transform(df_numeric_f), columns=df_numeric_f.columns)
    return df_imputed_m, df_imputed_f

# Fonction pour appliquer PCA et détecter les anomalies
def pca_and_filter_anomalies(df, n_components=2, eps=200, min_samples=2):
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(df)
    
    pca = PCA(n_components=n_components)
    data_pca = pca.fit_transform(data_scaled)
    
    # Reconstruction des données
    data_reconstructed = pca.inverse_transform(data_pca)
    reconstruction_error = np.mean((data_scaled - data_reconstructed) ** 2, axis=1)
    
    # Détection des anomalies avec DBSCAN
    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    clusters = dbscan.fit_predict(data_pca)
    
    # Marquer les anomalies
    anomalies = clusters == -1
    normal_data = df[~anomalies]
    
    return normal_data, reconstruction_error

# Fonction pour entraîner et sauvegarder le modèle
def train_and_save_model(csv_file_path, model_file_path_male, model_file_path_female):
    df_male, df_female = load_and_prepare_data(csv_file_path)
    df_male, _ = pca_and_filter_anomalies(df_male)
    df_female, _ = pca_and_filter_anomalies(df_female)
    
    # Diviser les données en 90% pour l'entraînement et 10% pour le test
    train_size_male = int(0.99 * len(df_male))
    train_size_female = int(0.99 * len(df_female))
    
    X_train_male = df_male[:train_size_male].to_numpy()
    X_train_female = df_female[:train_size_female].to_numpy()
    
    rf_male = RandomForestRegressor()
    rf_female = RandomForestRegressor()
    
    rf_male.fit(X_train_male, X_train_male)
    rf_female.fit(X_train_female, X_train_female)
    
    dump(rf_male, model_file_path_male)
    dump(rf_female, model_file_path_female)
    
    print(f"Modèle masculin sauvegardé dans {model_file_path_male}")
    print(f"Modèle féminin sauvegardé dans {model_file_path_female}")

# Fonction pour compléter un vecteur partiel
def complete_vector(partial_vector, model_file_path, index_to_replace):
    if not os.path.exists(model_file_path):
        raise FileNotFoundError(f"Le fichier du modèle '{model_file_path}' est introuvable. Veuillez entraîner le modèle d'abord.")
    
    rf = load(model_file_path)
    partial_vector = partial_vector.reshape(1, -1)
    missing_value = rf.predict(partial_vector)[0, index_to_replace]
    return missing_value

# Fonction pour évaluer les erreurs sur les derniers vecteurs
def evaluate_errors(df, model_file_path, start_index):
    errors_matrix = []
    
    for vector_index in range(start_index, len(df)):
        vector_row = df.iloc[vector_index].to_numpy()
        vector0 = vector_row.copy()
        print(f"Vecteur original {vector_index} :", vector_row)
        
        error_vector = []
        
        for i in range(len(vector_row)):
            partial_vector = vector_row.copy()
            partial_vector[i] = np.nan
            deduced_value = complete_vector(partial_vector, model_file_path, i)
            actual_value = vector0[i]
            error = (actual_value - deduced_value) / actual_value if actual_value != 0 else 0
            error_vector.append(error)
        
        errors_matrix.append(error_vector)
    
    errors_df = pd.DataFrame(errors_matrix, columns=coordinate_names[:len(errors_matrix[0])])
    return errors_df

# Fonction pour afficher et sauvegarder les graphiques des erreurs avec les écarts-types
def plot_errors_with_std(errors_df, title, filename):
    plt.figure(figsize=(10, 6))
    mean_errors = errors_df.mean()
    std_errors = errors_df.std()
    
    plt.errorbar(range(len(mean_errors)), mean_errors, yerr=std_errors, fmt='o', capsize=5, label='Erreur Moyenne avec Écart-Type')
    
    plt.xlabel('Coordonnées')
    plt.ylabel('Erreur')
    plt.title(title)
    plt.xticks(range(len(mean_errors)), errors_df.columns, rotation=90)
    plt.legend()
    plt.tight_layout()
    
    # Sauvegarder le graphique en tant qu'image
    plt.savefig(filename)
    plt.close()

# Fonction principale
def main():
    model_file_path_male = "random_forest_model_male.joblib"
    model_file_path_female = "random_forest_model_female.joblib"
    
    if len(sys.argv) > 1 and sys.argv[1] == 'train':
        train_and_save_model(csv_file_path, model_file_path_male, model_file_path_female)
    else:
        df_male, df_female = load_and_prepare_data(csv_file_path)
        df_male, _ = pca_and_filter_anomalies(df_male)
        df_female, _ = pca_and_filter_anomalies(df_female)
        
        # Diviser les données en 90% pour l'entraînement et 10% pour l'analyse
        train_size_male = int(0.99 * len(df_male))
        train_size_female = int(0.99 * len(df_female))
        
        test_data_male = df_male[train_size_male:]
        test_data_female = df_female[train_size_female:]
        
        # Évaluer les erreurs sur les données de test pour les hommes
        errors_df_male = evaluate_errors(test_data_male, model_file_path_male, 0)
        mean_errors_male = errors_df_male.mean()
        print("Erreurs moyennes pour chaque coordonnée (hommes) :")
        print(mean_errors_male)
        
        # Évaluer les erreurs sur les données de test pour les femmes
        errors_df_female = evaluate_errors(test_data_female, model_file_path_female, 0)
        mean_errors_female = errors_df_female.mean()
        print("Erreurs moyennes pour chaque coordonnée (femmes) :")
        print(mean_errors_female)
        
        # Afficher et sauvegarder les graphiques des erreurs avec les écarts-types
        plot_errors_with_std(errors_df_male, "Erreurs de Prédiction pour les Hommes", "errors_male.png")
        plot_errors_with_std(errors_df_female, "Erreurs de Prédiction pour les Femmes", "errors_female.png")

if __name__ == "__main__":
    main()
