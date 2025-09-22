"""
Time Series Stationarity Classification - Model Training Module
Fast and efficient model training with multiple algorithms
"""

import numpy as np
import pandas as pd
import pickle
import json
import os
import time
from typing import Dict, Tuple, Any
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, classification_report)
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.tree import DecisionTreeClassifier
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from sklearn.neural_network import MLPClassifier

import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import joblib
import gc

# Proje konfigürasyonlarını import et
from config import (PROCESSED_DATA_DIR, TRAINED_MODELS_DIR, REPORTS_DIR,
                    TEST_SIZE, FEATURE_SELECTION_K, CROSS_VALIDATION_FOLDS)

class StationarityModelTrainer:
    """Train multiple models for stationarity classification"""
    
    def __init__(self, data_dir: str):
        self.data_dir = data_dir
        self.models = {}
        self.scalers = {}
        self.results = {}
        self.best_model = None
        self.feature_names = None
        
    def load_data(self) -> Tuple[np.ndarray, np.ndarray]:
        """Load processed features and labels"""
        print("Loading processed data...")
        X = np.load(os.path.join(self.data_dir, 'features.npy'))
        y = np.load(os.path.join(self.data_dir, 'labels.npy'))
        
        feature_names_path = os.path.join(self.data_dir, 'feature_names.json')
        if os.path.exists(feature_names_path):
            with open(feature_names_path, 'r') as f:
                self.feature_names = json.load(f)
        
        print(f"Loaded data shape: X={X.shape}, y={y.shape}")
        print(f"Class distribution: {np.bincount(y)}")
        return X, y
    
    def preprocess_features(self, X: np.ndarray, method: str = 'robust') -> Tuple[np.ndarray, Any]:
        """Scale and preprocess features"""
        if method == 'standard': scaler = StandardScaler()
        elif method == 'robust': scaler = RobustScaler()
        else: return X, None
        return scaler.fit_transform(X), scaler
    
    def feature_selection(self, X: np.ndarray, y: np.ndarray, method: str = 'kbest', k: int = 30) -> Tuple[np.ndarray, Any]:
        """Select most important features"""
        k = min(k, X.shape[1])
        if method == 'kbest': selector = SelectKBest(f_classif, k=k)
        elif method == 'mutual_info': selector = SelectKBest(mutual_info_classif, k=k)
        elif method == 'pca': selector = PCA(n_components=k, random_state=42)
        else: return X, None
        return selector.fit_transform(X, y), selector
    
    def get_fast_models(self) -> Dict:
        """Get dictionary of fast training models"""
        return {
            'logistic_regression': LogisticRegression(max_iter=1000, random_state=42, n_jobs=-1, solver='saga'),
            'sgd_classifier': SGDClassifier(loss='log_loss', max_iter=1000, random_state=42, n_jobs=-1, early_stopping=True),
            'naive_bayes': GaussianNB(),
            'decision_tree': DecisionTreeClassifier(max_depth=10, min_samples_split=20, random_state=42),
            'random_forest_fast': RandomForestClassifier(n_estimators=100, max_depth=10, n_jobs=-1, random_state=42),
            'extra_trees': ExtraTreesClassifier(n_estimators=100, max_depth=10, n_jobs=-1, random_state=42),
            'lightgbm': LGBMClassifier(n_estimators=200, random_state=42, verbose=-1, n_jobs=-1),
            'xgboost_fast': XGBClassifier(n_estimators=200, random_state=42, tree_method='hist', verbosity=0, n_jobs=-1),
            'mlp_fast': MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=500, early_stopping=True, random_state=42)
        }
    
    def train_single_model(self, model, X_train, y_train, X_val, y_val, model_name: str) -> Dict:
        """Train a single model and evaluate"""
        print(f"\nTraining {model_name}...")
        start_time = time.time()
        model.fit(X_train, y_train)
        train_time = time.time() - start_time
        
        y_train_pred = model.predict(X_train)
        y_val_pred = model.predict(X_val)
        y_val_proba = model.predict_proba(X_val)[:, 1] if hasattr(model, 'predict_proba') else None
        
        results = {
            'model_name': model_name, 'train_time': train_time,
            'train_accuracy': accuracy_score(y_train, y_train_pred),
            'val_accuracy': accuracy_score(y_val, y_val_pred),
            'val_precision': precision_score(y_val, y_val_pred, average='weighted'),
            'val_recall': recall_score(y_val, y_val_pred, average='weighted'),
            'val_f1': f1_score(y_val, y_val_pred, average='weighted'),
            'val_roc_auc': roc_auc_score(y_val, y_val_proba) if y_val_proba is not None else None,
            'confusion_matrix': confusion_matrix(y_val, y_val_pred),
            'classification_report': classification_report(y_val, y_val_pred, output_dict=True)
        }
        
        print(f"{model_name} -> Val Acc: {results['val_accuracy']:.4f}, F1: {results['val_f1']:.4f}, Time: {train_time:.2f}s")
        return results
    
    def train_all_models(self, X: np.ndarray, y: np.ndarray, test_size: float, k_features: int):
        """Train all models and compare results"""
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42, stratify=y)
        
        X_train_scaled, scaler = self.preprocess_features(X_train, method='robust')
        X_test_scaled = scaler.transform(X_test)
        self.scalers['main'] = scaler
        
        if X_train.shape[1] > k_features:
            print(f"Performing feature selection to keep best {k_features} features.")
            X_train_selected, selector = self.feature_selection(X_train_scaled, y_train, method='kbest', k=k_features)
            X_test_selected = selector.transform(X_test_scaled)
            self.scalers['selector'] = selector
        else:
            X_train_selected, X_test_selected = X_train_scaled, X_test_scaled
        
        models_dict = self.get_fast_models()
        all_results = []
        for model_name, model in tqdm(models_dict.items(), desc="Training Models"):
            try:
                results = self.train_single_model(model, X_train_selected, y_train, X_test_selected, y_test, model_name)
                all_results.append(results)
                self.models[model_name] = model
                self.results[model_name] = results
            except Exception as e:
                print(f"Error training {model_name}: {e}")
            gc.collect()
        
        best_f1 = -1
        if all_results:
            for result in all_results:
                if result['val_f1'] > best_f1:
                    best_f1 = result['val_f1']
                    self.best_model = result['model_name']
        
        if self.best_model:
            print(f"\n{'='*50}\nBest model: {self.best_model} with F1 score: {best_f1:.4f}\n{'='*50}")
        return all_results
    
    def cross_validate_best_model(self, X: np.ndarray, y: np.ndarray, cv: int):
        """Perform cross-validation on the best model"""
        if not self.best_model:
            print("No best model found.")
            return
        
        print(f"\nCross-validating {self.best_model}...")
        X_scaled, _ = self.preprocess_features(X, method='robust')
        model = self.get_fast_models()[self.best_model]
        cv_scores = cross_val_score(model, X_scaled, y, cv=cv, scoring='f1_weighted', n_jobs=-1)
        print(f"CV F1 scores: {cv_scores}")
        print(f"Mean CV F1: {np.mean(cv_scores):.4f} (+/- {np.std(cv_scores) * 2:.4f})")
        return cv_scores
    
    def save_models(self, output_dir: str):
        """Save trained models and preprocessing objects"""
        os.makedirs(output_dir, exist_ok=True)
        for model_name, model in self.models.items():
            joblib.dump(model, os.path.join(output_dir, f'{model_name}.joblib'))
        
        with open(os.path.join(output_dir, 'scalers.pkl'), 'wb') as f:
            pickle.dump(self.scalers, f)
        
        json_results = {}
        for key, result in self.results.items():
            json_result = result.copy()
            if 'confusion_matrix' in json_result:
                json_result['confusion_matrix'] = json_result['confusion_matrix'].tolist()
            json_results[key] = json_result
        with open(os.path.join(output_dir, 'training_results.json'), 'w') as f:
            json.dump(json_results, f, indent=2)
        
        best_model_info = {'best_model': self.best_model, 'best_f1': self.results[self.best_model]['val_f1'], 'feature_names': self.feature_names}
        with open(os.path.join(output_dir, 'best_model_info.json'), 'w') as f:
            json.dump(best_model_info, f, indent=2)
        print(f"\nAll models and results saved to {output_dir}")
    
    def plot_results(self, save_path: str):
        """Plot model comparison results"""
        if not self.results: return
        
        metrics_data = [{'Model': r['model_name'], 'F1 Score': r['val_f1'], 'Training Time (s)': r['train_time']} for r in self.results.values()]
        df = pd.DataFrame(metrics_data).sort_values('F1 Score', ascending=False)
        
        fig, axes = plt.subplots(1, 2, figsize=(16, 7))
        sns.barplot(x='F1 Score', y='Model', data=df, ax=axes[0], palette='viridis')
        axes[0].set_title('Model F1 Score Comparison')
        
        sns.barplot(x='Training Time (s)', y='Model', data=df, ax=axes[1], palette='plasma')
        axes[1].set_title('Model Training Time Comparison')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300)
        plt.show()
        print(f"Results plot saved to {save_path}")

def run_training():
    """Model eğitim pipeline'ını çalıştırır."""
    print("\n--- Model Eğitimi Aşaması Başladı ---")
    trainer = StationarityModelTrainer(data_dir=str(PROCESSED_DATA_DIR))
    
    try:
        X, y = trainer.load_data()
    except FileNotFoundError:
        print(f"Hata: İşlenmiş veri '{PROCESSED_DATA_DIR}' klasöründe bulunamadı.")
        print("Lütfen önce veri işleme adımını çalıştırdığınızdan emin olun.")
        return

    trainer.train_all_models(X, y, test_size=TEST_SIZE, k_features=FEATURE_SELECTION_K)
    if trainer.best_model:
        trainer.cross_validate_best_model(X, y, cv=CROSS_VALIDATION_FOLDS)
        trainer.save_models(output_dir=str(TRAINED_MODELS_DIR))
        reports_path = REPORTS_DIR / "model_comparison.png"
        os.makedirs(REPORTS_DIR, exist_ok=True)
        trainer.plot_results(save_path=str(reports_path))
    else:
        print("Hiçbir model başarıyla eğitilemedi.")
    
    print("--- Model Eğitimi Aşaması Tamamlandı ---")

if __name__ == "__main__":
    run_training()