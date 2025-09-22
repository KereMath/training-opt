import os
import json
import sys
import multiprocessing
from pathlib import Path
from collections import defaultdict

# Proje ana dizinini Python'un modül arama yoluna ekle
project_root = Path(__file__).resolve().parent
sys.path.append(str(project_root))

from predictor import Predictor
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from tqdm import tqdm

# --- AYARLAR ---

# Test veri setlerinin bulunduğu ana klasör
TEST_DATA_ROOT = Path(r'C:\Users\user\Desktop\test datasets')

# Sonuçların kaydedileceği JSON dosyasının tam yolu
OUTPUT_FILE = TEST_DATA_ROOT / 'evaluation_results.json'

# KULLANILACAK İŞLEMCİ ÇEKİRDEĞİ SAYISI
# 24 çekirdeğiniz varsa, sistemin kararlılığı için birkaçını boşta bırakmak iyidir.
# Bu sayıyı kendi sisteminize göre ayarlayabilirsiniz.
NUM_WORKERS = 22 # os.cpu_count() - 2 gibi bir değer idealdir.

# Etiketlerin modeldeki karşılıkları
LABEL_MAPPING = {'stationary': 0, 'non_stationary': 1}
CLASS_NAMES = list(LABEL_MAPPING.keys())

# --- MULTIPROCESSING İÇİN YARDIMCI FONKSİYONLAR ---

# Global predictor nesnesi, her işçi proses tarafından bir kez oluşturulacak.
predictor_instance = None

def init_worker():
    """
    Her işçi proses başladığında bu fonksiyon bir kez çalışır.
    Büyük Predictor nesnesini ve modelleri her iş için tekrar tekrar yüklemek yerine
    sadece bir kez yükleyerek performansı artırır.
    """
    global predictor_instance
    print(f"İşçi Proses {os.getpid()} modelleri yüklüyor...")
    # Her işçi kendi Predictor kopyasını oluşturur ve bellekte tutar.
    predictor_instance = Predictor()
    print(f"İşçi Proses {os.getpid()} hazır.")

def get_true_label(file_path: Path, root_dir: Path) -> str:
    """Dosya yoluna göre gerçek etiketi belirler."""
    if 'non stationary sets' in str(file_path.relative_to(root_dir)):
        return 'non_stationary'
    elif 'stationary sets' in str(file_path.relative_to(root_dir)):
        return 'stationary'
    return None

def process_single_file(file_path: Path):
    """
    Tek bir CSV dosyasını işleyen ve sonucu döndüren işçi fonksiyonu.
    Bu fonksiyon, işlem havuzundaki her bir işçi tarafından çalıştırılır.
    """
    global predictor_instance
    true_label = get_true_label(file_path, TEST_DATA_ROOT)
    
    if not true_label or not predictor_instance:
        return None

    try:
        prediction_results = predictor_instance.predict(str(file_path))
        
        if "error" in prediction_results:
            # Hata durumunda dosya adını ve hatayı döndür
            return (file_path.name, {"error": prediction_results['error']})
            
        # Sadece model adları ve tahminlerini içeren bir sözlük oluştur
        predictions = {p['model_name']: p['prediction'] for p in prediction_results['all_predictions']}
        return (true_label, predictions)
    except Exception as e:
        return (file_path.name, {"error": str(e)})


def main():
    """Ana değerlendirme fonksiyonu."""
    print("Çoklu işlemci ile değerlendirme süreci başlatılıyor...")
    
    # Geçici olarak bir Predictor oluşturup model isimlerini alıyoruz.
    try:
        temp_predictor = Predictor()
        model_names = list(temp_predictor.models.keys())
        del temp_predictor # Hafızadan temizle
        print(f"\n{len(model_names)} model bulundu: {', '.join(model_names)}")
    except Exception as e:
        print(f"HATA: Modeller yüklenemedi. Hata: {e}")
        return

    all_files = list(TEST_DATA_ROOT.rglob('*.csv'))
    print(f"Toplam {len(all_files)} adet CSV dosyası bulundu.")
    if not all_files:
        return

    true_labels = []
    model_predictions = defaultdict(list)

    # --- İŞLEM HAVUZU (PROCESSING POOL) ---
    print(f"\n{NUM_WORKERS} adet işçi proses ile işlem havuzu oluşturuluyor...")
    # `with` bloğu, havuzun iş bitince düzgünce kapanmasını sağlar.
    with multiprocessing.Pool(processes=NUM_WORKERS, initializer=init_worker) as pool:
        # `imap_unordered` görevleri işçilere dağıtır ve sonuçları geldikçe döndürür.
        # Bu, tqdm ile ilerlemeyi anlık olarak takip etmemizi sağlar.
        results_iterator = pool.imap_unordered(process_single_file, all_files)
        
        print("\nTüm dosyalar için tahminler yapılıyor...")
        # Gelen sonuçları işle
        for result in tqdm(results_iterator, total=len(all_files), desc="Dosyalar işleniyor"):
            if result is None:
                continue
            
            label, predictions = result
            
            if "error" in predictions:
                tqdm.write(f"HATA: '{label}' işlenirken hata oluştu: {predictions['error']}")
                continue
            
            true_labels.append(label)
            for model_name in model_names:
                model_predictions[model_name].append(predictions.get(model_name, "ERROR"))

    # --- METRİK HESAPLAMA (Bu kısım aynı kaldı) ---
    print("\nTüm modeller için metrikler hesaplanıyor...")
    final_evaluation = {}
    for model_name in model_names:
        y_true = true_labels
        y_pred = model_predictions[model_name]
        
        filtered_true = [label for i, label in enumerate(y_true) if y_pred[i] != "ERROR"]
        filtered_pred = [pred for pred in y_pred if pred != "ERROR"]
        
        if not filtered_pred:
            print(f"Uyarı: '{model_name}' için hiç başarılı tahmin bulunamadı.")
            continue

        print(f"\n--- Model: {model_name} ---")
        cm = confusion_matrix(filtered_true, filtered_pred, labels=CLASS_NAMES)
        report = classification_report(filtered_true, filtered_pred, target_names=CLASS_NAMES, output_dict=True)
        accuracy = accuracy_score(filtered_true, filtered_pred)
        
        print(f"Doğruluk (Accuracy): {accuracy:.4f}")
        print(classification_report(filtered_true, filtered_pred, target_names=CLASS_NAMES))
        print("Karmaşıklık Matrisi:\n", cm)
        
        final_evaluation[model_name] = {
            'accuracy': accuracy,
            'classification_report': report,
            'confusion_matrix': {'labels': CLASS_NAMES, 'matrix': cm.tolist()},
            'total_samples_tested': len(filtered_true)
        }
        
    print(f"\nDeğerlendirme sonuçları '{OUTPUT_FILE}' dosyasına kaydediliyor...")
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        json.dump(final_evaluation, f, ensure_ascii=False, indent=4)
    print("✅ İşlem başarıyla tamamlandı!")

if __name__ == '__main__':
    # Bu satır, multiprocessing'in Windows'ta doğru çalışması için zorunludur.
    multiprocessing.freeze_support()
    main()