"""
Ana Çalıştırıcı Betik
Bu betik, veri işleme ve model eğitimi adımlarını sırayla çalıştırır.
"""
from processor import run_processing
from trainer import run_training

def main():
    """Projenin tüm adımlarını çalıştırır."""
    # 1. Adım: Verileri işle, özellikleri çıkar ve kaydet.
    run_processing()
    
    # 2. Adım: İşlenmiş verilerle modelleri eğit, değerlendir ve kaydet.
    run_training()

if __name__ == "__main__":
    main()