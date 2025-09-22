"""
Proje genelindeki tüm konfigürasyonları ve sabitleri içerir.
"""
from pathlib import Path

# --- TEMEL DOSYA YOLLARI ---
BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = r"C:\Users\user\Desktop\LASTDATA"  # Ham verilerin bulunduğu ana klasör
PROCESSED_DATA_DIR = BASE_DIR / "processed_data"
TRAINED_MODELS_DIR = BASE_DIR / "trained_models"
REPORTS_DIR = BASE_DIR / "reports"
UPLOAD_DIR = BASE_DIR / "uploads" # Flask uygulaması için

# --- VERİ İŞLEME PARAMETRELERİ ---
CHUNK_SIZE = 10000  # Bellek dostu okuma için chunk boyutu

# !!! YENİ EKLENEN AYAR !!!
# Test için her bir ana klasörden (stationary, collective_anomaly, vb.) alınacak maksimum dosya sayısı.
# Tüm veriyi işlemek için bu değeri None yapın.
FILES_PER_FOLDER_LIMIT = None

# SAMPLE_SIZE parametresini artık kullanmıyoruz, bu yeni yöntem daha dengeli bir test seti oluşturur.
# SAMPLE_SIZE = None

# --- ETİKETLEME ---
# DİKKAT: Kodunuzda stationary=0, non-stationary=1 olarak ayarlanmış.
# Bu haritalama ona uygun yapılmıştır.
LABEL_MAP = {
    'stationary': 0,
    'non_stationary': 1
}

# --- MODEL EĞİTİM PARAMETRELERİ ---
TEST_SIZE = 0.2  # Test verisi oranı
FEATURE_SELECTION_K = 30  # En iyi K adet özellik seçimi
CROSS_VALIDATION_FOLDS = 5 # Çapraz doğrulama kat sayısı