
# ==============================================================================
# PROJECT: Income Classification Pipeline (R-to-Python Refactor)
# AUTHOR: Mehmet Ali Erkan
# DATE: 2025
# ==============================================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score

# 1. DATA LOADING & CLEANING
# ------------------------------------------------------------------------------
def load_and_clean_data(filepath):
    """Loads data and applies specific cleaning rules from the original study."""
    print(f"[INFO] Loading raw data from {filepath}...")
    
    # R kodundaki "read.csv" ve "trimws" karşılığı
    df = pd.read_csv(filepath, skipinitialspace=True)
    
    # R'da yapılan: data[data == "?"] <- NA
    df.replace('?', np.nan, inplace=True)
    
    print(f"[INFO] Missing values before cleaning:\n{df.isnull().sum()}")

    # R Notebook'undaki Özel Gruplamalar (Feature Engineering)
    # Workclass: Government birleştirme
    df['workclass'] = df['workclass'].replace(
        ['Federal-gov', 'Local-gov', 'State-gov'], 'Government'
    )
    df['workclass'] = df['workclass'].replace(
        ['Self-emp-inc', 'Self-emp-not-inc'], 'Self_employment'
    )
    
    # Marital Status: Evlileri birleştirme
    df['marital-status'] = df['marital-status'].replace(
        ['Married-AF-spouse', 'Married-civ-spouse'], 'Married'
    )
    
    # Target Encoding (salary <=50K -> 0, >50K -> 1)
    df['salary'] = df['salary'].apply(lambda x: 1 if '>50K' in str(x) else 0)
    
    print("[INFO] Custom grouping and cleaning applied.")
    return df

# 2. PIPELINE CONSTRUCTION
# ------------------------------------------------------------------------------
def get_pipeline(X):
    """Constructs the preprocessing pipeline."""
    
    # Sütun tiplerini ayır
    numeric_features = X.select_dtypes(include=['int64', 'float64']).columns
    categorical_features = X.select_dtypes(include=['object']).columns

    # Sayısal: KNN Imputation (R'daki VIM kütüphanesi mantığıyla) + Scaling
    numeric_transformer = Pipeline(steps=[
        ('imputer', KNNImputer(n_neighbors=5)), 
        ('scaler', StandardScaler())
    ])

    # Kategorik: Eksikleri 'Missing' diye doldur + OneHot
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value='Missing')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ])
    
    return preprocessor

# 3. TRAINING & EVALUATION
# ------------------------------------------------------------------------------
def run_benchmark(X_train, X_test, y_train, y_test, preprocessor):
    """Trains multiple models and prints the results."""
    
    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "Decision Tree": DecisionTreeClassifier(max_depth=10),
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42)
    }
    
    results = {}
    
    for name, model in models.items():
        # Pipeline oluştur: Preprocessing -> Model
        clf = Pipeline(steps=[('preprocessor', preprocessor),
                              ('classifier', model)])
        
        print(f"\nTraining {name}...")
        clf.fit(X_train, y_train)
        
        y_pred = clf.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        results[name] = acc
        
        print(f"--> {name} Accuracy: {acc:.4f}")
        print(classification_report(y_test, y_pred))
        
    return results

# 4. MAIN EXECUTION
# ------------------------------------------------------------------------------
if __name__ == "__main__":
    DATA_PATH = 'salary.csv' 
    
    try:
        # A. Load Data
        df = load_and_clean_data(DATA_PATH)
        
        # B. Split Data
        X = df.drop('salary', axis=1)
        y = df['salary']
        
        # R'daki "createDataPartition" karşılığı
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
        
        # C. Build Pipeline
        preprocessor = get_pipeline(X)
        
        # D. Run Benchmark
        print("\n[INFO] Starting Model Benchmark...")
        results = run_benchmark(X_train, X_test, y_train, y_test, preprocessor)
        
        print("\n--- FINAL RESULTS ---")
        for model, score in results.items():
            print(f"{model}: {score:.4f}")
            
    except FileNotFoundError:
        print(f"Error: '{DATA_PATH}' not found. Please place the CSV file in the same directory.")