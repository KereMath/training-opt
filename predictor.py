"""
Inference (Çıkarım) Modülü
Eğitilmiş TÜM modelleri kullanarak tahmin yapar ve sonuçları karşılaştırır.
En iyi modeli vurgular, tüm model sonuçlarını gösterir.
"""
import pandas as pd
import numpy as np
import joblib
import json
import pickle
from pathlib import Path
from typing import Dict, Any, List
import os

from config import TRAINED_MODELS_DIR, CHUNK_SIZE, LABEL_MAP
from processor import TimeSeriesDataProcessor

class Predictor:
    def __init__(self, model_dir: Path = TRAINED_MODELS_DIR):
        self.model_dir = model_dir
        if not (self.model_dir / 'best_model_info.json').exists():
            raise FileNotFoundError(f"Best model info not found in {self.model_dir}. Please train models first.")
        
        # En iyi model bilgisini yükle
        with open(self.model_dir / 'best_model_info.json', 'r') as f:
            best_model_info = json.load(f)
        self.best_model_name = best_model_info['best_model']
        self.best_model_score = best_model_info.get('best_score', 0.0)
        
        # Tüm modelleri yükle
        self.models = {}
        self.load_all_models()
        
        # Ön işleme araçlarını yükle
        with open(self.model_dir / 'scalers.pkl', 'rb') as f:
            self.scalers = pickle.load(f)
        
        self.main_scaler = self.scalers.get('main')
        self.selector = self.scalers.get('selector')
        self.feature_extractor = TimeSeriesDataProcessor(base_path='', chunk_size=CHUNK_SIZE)
        self.inverse_label_map = {v: k for k, v in LABEL_MAP.items()}

    def load_all_models(self):
        """Tüm eğitilmiş modelleri yükle"""
        model_files = list(self.model_dir.glob('*.joblib'))
        
        for model_file in model_files:
            model_name = model_file.stem
            try:
                model = joblib.load(model_file)
                self.models[model_name] = model
                print(f"Model loaded: {model_name}")
            except Exception as e:
                print(f"Error loading model {model_name}: {e}")

    def predict_single_model(self, selected_features: np.ndarray, model_name: str, model) -> Dict[str, Any]:
        """Tek bir model için tahmin yap"""
        try:
            prediction_idx = model.predict(selected_features)[0]
            prediction_label = self.inverse_label_map[prediction_idx]
            
            # Güven skorları
            confidence_scores = {}
            if hasattr(model, 'predict_proba'):
                probabilities = model.predict_proba(selected_features)[0]
                confidence_scores = {
                    self.inverse_label_map[i]: round(float(prob), 4) 
                    for i, prob in enumerate(probabilities)
                }
                max_confidence = round(float(max(probabilities)), 4)
            else:
                max_confidence = "N/A"
            
            return {
                "model_name": model_name,
                "prediction": prediction_label,
                "confidence_scores": confidence_scores,
                "max_confidence": max_confidence,
                "is_best": model_name == self.best_model_name
            }
        except Exception as e:
            return {
                "model_name": model_name,
                "prediction": "ERROR",
                "error": str(e),
                "confidence_scores": {},
                "max_confidence": "N/A",
                "is_best": model_name == self.best_model_name
            }

    def predict(self, csv_path: str) -> Dict[str, Any]:
        """Tüm modellerden tahmin al"""
        try:
            # Özellik çıkarımı
            result = self.feature_extractor.process_single_file(Path(csv_path), label=-1)
            if result is None:
                return {"error": "Could not extract features from the file."}
            
            feature_vector, _ = result
            
            # Özellik boyutunu ayarla
            expected_features = self.main_scaler.n_features_in_
            if len(feature_vector) != expected_features:
                if len(feature_vector) < expected_features:
                    feature_vector = np.pad(feature_vector, (0, expected_features - len(feature_vector)), 'constant')
                else:
                    feature_vector = feature_vector[:expected_features]
            
            feature_vector = feature_vector.reshape(1, -1)
            
            # Ön işleme
            scaled_features = self.main_scaler.transform(feature_vector)
            selected_features = self.selector.transform(scaled_features) if self.selector else scaled_features
            
            # Tüm modellerden tahmin al
            all_predictions = []
            for model_name, model in self.models.items():
                prediction_result = self.predict_single_model(selected_features, model_name, model)
                all_predictions.append(prediction_result)
            
            # En iyi modelin tahminini bul
            best_prediction = next((p for p in all_predictions if p["is_best"]), None)
            
            # Sonuçları sırala (en iyi model en üstte, sonra güven skoruna göre)
            all_predictions.sort(key=lambda x: (not x["is_best"], -(x["max_confidence"] if isinstance(x["max_confidence"], float) else 0)))
            
            # Özet istatistikler
            stationary_votes = sum(1 for p in all_predictions if p["prediction"] == "stationary")
            non_stationary_votes = sum(1 for p in all_predictions if p["prediction"] == "non_stationary")
            total_models = len(all_predictions)
            
            consensus = "stationary" if stationary_votes > non_stationary_votes else "non_stationary"
            consensus_percentage = round(max(stationary_votes, non_stationary_votes) / total_models * 100, 1)
            
            return {
                "file_name": Path(csv_path).name,
                "best_model_prediction": best_prediction,
                "all_predictions": all_predictions,
                "summary": {
                    "consensus": consensus,
                    "consensus_percentage": consensus_percentage,
                    "stationary_votes": stationary_votes,
                    "non_stationary_votes": non_stationary_votes,
                    "total_models": total_models
                },
                "best_model_info": {
                    "name": self.best_model_name,
                    "score": self.best_model_score
                }
            }
            
        except Exception as e:
            return {"error": f"An error occurred during prediction: {str(e)}"}