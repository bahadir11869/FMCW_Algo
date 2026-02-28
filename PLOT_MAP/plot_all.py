import os
import numpy as np
import matplotlib.pyplot as plt

NUM_CHIRPS = 256
NUM_SAMPLES = 128

def load_radar_data(filename, expected_shape=(NUM_CHIRPS, NUM_SAMPLES)):
    try:
        data = np.genfromtxt(filename, delimiter=',')
        data = np.array(data)

        if data.ndim == 0:
            return None

        if data.ndim == 1:
            N = expected_shape[0] * expected_shape[1]
            if data.size == N:
                data = data.reshape(expected_shape)
            else:
                print(f"[UYARI] {filename} 1D geldi ama boyut uyuşmuyor: {data.size} != {N}")
                return None

        if data.shape != expected_shape:
            if data.shape == (expected_shape[1], expected_shape[0]):
                data = data.T
            else:
                print(f"[UYARI] {filename} shape beklenenden farklı: {data.shape} != {expected_shape}")
                return None

        return data
    except Exception as e:
        print(f"[HATA] {filename} okunamadı: {e}")
        return None


def plot_in_new_window(title_prefix, power_map, mask_map):
    # KRİTİK DEĞİŞİKLİK 1: Doppler eksenini (0. eksen - satırlar) ortaya kaydır (Shift)
    # Bu sayede 0 hızı grafiğin ortasına gelir.
    #shifted_power_map = np.fft.fftshift(power_map, axes=0)
    #shifted_mask_map = np.fft.fftshift(mask_map, axes=0)
    
    # Gücü dB cinsine çevir
    power_db = 10 * np.log10(power_map + 1e-9)

    fig = plt.figure(figsize=(12, 6))
    if hasattr(fig.canvas.manager, 'set_window_title'):
        fig.canvas.manager.set_window_title(title_prefix)

    # KRİTİK DEĞİŞİKLİK 2: Eksen değerlerini (extent) tanımla
    # x ekseni (Range): 0'dan SAMPLES'a
    # y ekseni (Doppler): -CHIRPS/2'den +CHIRPS/2'ye
    doppler_extent = [-NUM_CHIRPS // 2, NUM_CHIRPS // 2]
    range_extent = [0, NUM_SAMPLES]
    img_extent = [range_extent[0], range_extent[1], doppler_extent[0], doppler_extent[1]]

    # 1. Subplot: Sadece RDM
    ax1 = fig.add_subplot(1, 2, 1)
    im = ax1.imshow(power_db, aspect='auto', cmap='jet', origin='lower', extent=img_extent)
    ax1.set_title(f"{title_prefix} | Range-Doppler (dB)")
    ax1.set_xlabel("Range (Sample / Mesafe)")
    ax1.set_ylabel("Doppler (Chirp / Hız)")
    fig.colorbar(im, ax=ax1, label="Güç (dB)")

    # 2. Subplot: RDM üzerinde CFAR Hedefleri
    ax2 = fig.add_subplot(1, 2, 2)
    ax2.imshow(power_db, aspect='auto', cmap='gray', origin='lower', alpha=0.5, extent=img_extent)
    ax2.set_title(f"{title_prefix} | CFAR Mask Tespitleri")
    ax2.set_xlabel("Range (Sample / Mesafe)")
    ax2.set_ylabel("Doppler (Chirp / Hız)")

    # Hedefleri bul (shifted map üzerinden)
    targets = np.argwhere(mask_map > 0.5)
    
    if len(targets) > 0:
        # Koordinatları görselin extent sınırlarına göre uyarla
        # argwhere bize [Doppler_Index, Range_Index] döner
        target_range = targets[:, 1]
        target_doppler = targets[:, 0] - (NUM_CHIRPS // 2) # Y eksenini sıfır merkezli yap

        ax2.scatter(
            target_range,
            target_doppler,
            s=40,
            marker='x',
            color='red',
            linewidths=2,
            label='Tespit'
        )
        ax2.legend(loc="upper right")
    else:
        ax2.text(0.5, 0.5, "Hedef bulunamadı", transform=ax2.transAxes, ha='center', va='center')

    fig.tight_layout()
    plt.savefig(title_prefix + ".png")


def main():
    base = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

    candidates = [
        ("CPU_Recursive",  os.path.join(base, "CPU_FMCW", "FFT_CPU.csv"), os.path.join(base, "CPU_FMCW", "CFAR_CPU.csv")),
        ("CPU_AVX",        os.path.join(base, "CPU_FMCW", "FFT_AVX.csv"), os.path.join(base, "CPU_FMCW", "CFAR_AVX.csv")),
        ("GPU_2DFFT",      os.path.join(base, "GPU_FMCW", "FFT_2DFFT.csv"), os.path.join(base, "GPU_FMCW", "CFAR_2DFFT.csv")),
        ("GPU_ManuelSHM",  os.path.join(base, "GPU_FMCW", "FFT_ManuelSHM.csv"), os.path.join(base, "GPU_FMCW", "CFAR_ManuelSHM.csv")),
    ]

    opened_any = False

    for name, power_path, mask_path in candidates:
        if not (os.path.exists(power_path) and os.path.exists(mask_path)):
            print(f"[SKIP] {name}: dosyalar yok -> {power_path} / {mask_path}")
            continue

        power_map = load_radar_data(power_path)
        mask_map  = load_radar_data(mask_path)

        if power_map is None or mask_map is None:
            print(f"[SKIP] {name}: okunamadı/shape sorunu.")
            continue

        plot_in_new_window(name, power_map, mask_map)
        opened_any = True

    if not opened_any:
        print("Çizilecek veri bulunamadı. CSV yollarını ve shape'i kontrol et.")
        return

    plt.show()

if __name__ == "__main__":
    main()