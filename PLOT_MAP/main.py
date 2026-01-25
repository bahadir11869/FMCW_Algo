import numpy as np
import matplotlib.pyplot as plt
import os

# --- C++ defines.h ile Aynı Parametreler ---
NUM_CHIRPS = 256      
NUM_SAMPLES = 128     

C = 3e8
FC = 77e9
BANDWIDTH = 150e6
CHIRP_DURATION = 50e-6
FRAME_DURATION = NUM_CHIRPS * CHIRP_DURATION

# Çözünürlük Hesapları
RANGE_RES = C / (2 * BANDWIDTH)
VELOCITY_RES = (C / FC) / (2 * FRAME_DURATION)

def create_figure_for_file(filename, custom_title="RDM Plot"):
    """
    Verilen dosya için YENİ bir pencere açar ve çizer.
    """
    # Dosya var mı kontrolü
    if not os.path.exists(filename):
        print(f"UYARI: Dosya bulunamadı -> {filename}")
        return

    try:
        # CSV formatı: [Doppler x Range] 
        data = np.loadtxt(filename, delimiter=',')
    except Exception as e:
        print(f"HATA: {filename} okunamadı. ({e})")
        return

    if data.shape != (NUM_CHIRPS, NUM_SAMPLES):
        print(f"BOYUT HATASI: {filename} -> {data.shape}")
        return
    
    # 1. Shift ve dB Dönüşümü
    data_shifted = np.fft.fftshift(data, axes=0)
    data_db = 20 * np.log10(np.abs(data_shifted) + 1e-9)

    # 2. YENİ PENCERE OLUŞTURMA (Her çağrıldığında yeni ID alır)
    # figure() fonksiyonuna num=None verilirse her zaman yeni pencere açar.
    fig = plt.figure(figsize=(10, 6)) 
    
    # Pencere başlığını ayarla (Windows taskbar'da görünen isim)
    fig.canvas.manager.set_window_title(custom_title)

    # 3. Eksen Limitleri
    max_range = NUM_SAMPLES * RANGE_RES
    max_vel = (NUM_CHIRPS / 2) * VELOCITY_RES
    extent_vals = [0, max_range, -max_vel, max_vel]

    # 4. Çizim
    plt.imshow(data_db, aspect='auto', cmap='jet', origin='lower', extent=extent_vals)
    
    plt.title(f"{custom_title}\n(Res: {RANGE_RES:.2f} m, {VELOCITY_RES:.2f} m/s)")
    plt.xlabel("Range (m)")
    plt.ylabel("Velocity (m/s)")
    plt.colorbar(label='Amplitude (dB)')
    plt.grid(alpha=0.3)
    
    # İsteğe bağlı: Her pencere için PNG kaydet
    save_name = filename.replace('.csv', '.png')
    plt.savefig(save_name, dpi=150)
    print(f"Açıldı ve Kaydedildi: {custom_title} -> {save_name}")

if __name__ == "__main__":
    
    # --- BURAYA İSTEDİĞİN KADAR DOSYA EKLE ---
    # Format: ("Dosya Yolu", "Grafik Başlığı")
    files_to_plot = [
        ("../GPU_FMCW/2DFFT_GPU.csv", "1. Method: 2D FFT GPU cuFFT"),
        ("../CPU_FMCW/AVXFFT_CPU.csv",     "2. Method: CPU AVX Lib"),
    ]

    print("Grafikler hazırlanıyor...")

    # Listeyi dön ve her biri için fonksiyonu çağır
    for f_path, f_title in files_to_plot:
        create_figure_for_file(f_path, f_title)

    # Tüm pencereleri ekranda tut (Kod burada bloklanır)
    print("Tüm pencereler açık. Kapatmak için pencereleri kapatın.")
    plt.show()
