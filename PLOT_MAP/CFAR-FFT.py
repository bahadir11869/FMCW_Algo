import numpy as np
import matplotlib.pyplot as plt

# Ayarlar
NUM_CHIRPS = 256  # Doppler (Y ekseni)
NUM_SAMPLES = 128 # Range (X ekseni)

def load_radar_data(filename):
    # CSV formatındaki dosyayı oku
    try:
        data = np.loadtxt(filename, delimiter=',')
        return data
    except:
        print(f"{filename} okunamadı!")
        return None

# 1. Verileri Yükle
power_map = load_radar_data("../GPU_FMCW/2DFFT_GPU.csv")
mask_map = load_radar_data("../GPU_FMCW/radar_mask.csv")

if power_map is not None and mask_map is not None:
    
    # 2. Logaritmik Dönüşüm (dB) - Gürültüyü ve Hedefleri daha iyi görmek için
    # 0 olan değerlerde hata almamak için epsilon ekliyoruz
    power_db = 10 * np.log10(power_map + 1e-9)

    # 3. Görselleştirme
    plt.figure(figsize=(12, 6))

    # --- SOL TARAFA: Ham Radar Görüntüsü (Isı Haritası) ---
    plt.subplot(1, 2, 1)
    plt.title("Ham Range-Doppler Haritası (dB)")
    # aspect='auto' görüntüyü kareye zorlamaz
    plt.imshow(power_db, aspect='auto', cmap='jet', origin='lower')
    plt.colorbar(label='Güç (dB)')
    plt.xlabel('Range (Sample)')
    plt.ylabel('Doppler (Chirp)')

    # --- SAĞ TARAFA: CFAR Sonucu (Elenenler vs Seçilenler) ---
    plt.subplot(1, 2, 2)
    plt.title("CFAR Sonrası: Kırmızı Noktalar = HEDEF")
    
    # Önce arka plana sönük bir şekilde ham veriyi koyalım (Referans olsun)
    plt.imshow(power_db, aspect='auto', cmap='gray', origin='lower', alpha=0.5)
    
    # Üstüne CFAR'ın bulduğu hedefleri (Maskeyi) kırmızı noktalarla basalım
    # mask_map == 1 olan yerlerin koordinatlarını bul
    targets = np.argwhere(mask_map > 0.5) 
    
    # Scatter plot ile işaretle (x=Range, y=Doppler)
    # argwhere (satır, sütun) döner, scatter (x, y) ister -> (sütun, satır) verilir.
    if len(targets) > 0:
        plt.scatter(targets[:, 1], targets[:, 0], color='red', s=10, marker='x', label='Tespit Edilen')
    else:
        print("Hiç hedef bulunamadı!")

    plt.legend()
    plt.xlabel('Range (Sample)')
    plt.ylabel('Doppler (Chirp)')

    plt.tight_layout()
    plt.show()