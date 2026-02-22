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

        # 1D -> reshape dene
        if data.ndim == 1:
            N = expected_shape[0] * expected_shape[1]
            if data.size == N:
                data = data.reshape(expected_shape)
            else:
                print(f"[UYARI] {filename} 1D geldi ama boyut uyuşmuyor: {data.size} != {N}")
                return None

        # 2D shape kontrol
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
    power_db = 10 * np.log10(power_map + 1e-9)

    # Bu fonksiyon 1 algoritma için 1 FIGURE (yeni pencere) açar
    fig = plt.figure(figsize=(12, 6))
    fig.canvas.manager.set_window_title(title_prefix)  # bazı backendlerde çalışır

    ax1 = fig.add_subplot(1, 2, 1)
    im = ax1.imshow(power_db, aspect='auto', cmap='jet', origin='lower')
    ax1.set_title(f"{title_prefix} | Range-Doppler (dB)")
    ax1.set_xlabel("Range (Sample)")
    ax1.set_ylabel("Doppler (Chirp)")
    fig.colorbar(im, ax=ax1, label="Güç (dB)")

    ax2 = fig.add_subplot(1, 2, 2)
    ax2.imshow(power_db, aspect='auto', cmap='gray', origin='lower', alpha=0.5)
    ax2.set_title(f"{title_prefix} | CFAR Mask")
    ax2.set_xlabel("Range (Sample)")
    ax2.set_ylabel("Doppler (Chirp)")

    targets = np.argwhere(mask_map > 0.5)
    if len(targets) > 0:
        ax2.scatter(
    targets[:, 1],
    targets[:, 0],
    s=40,
    marker='x',
    color='red',
    linewidths=2,
    label='Tespit'
)

        ax2.legend(loc="upper right")
    else:
        ax2.text(0.5, 0.5, "Hedef bulunamadı", transform=ax2.transAxes,
                 ha='center', va='center')

    fig.tight_layout()
    plt.savefig(title_prefix + ".png")


def main():
    # python_all.py plot klasöründeyse bir üst dizine çıkıyor
    base = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

    candidates = [
        ("CPU Recursive",  os.path.join(base, "CPU_FMCW", "FFT_CPU.csv"),
                           os.path.join(base, "CPU_FMCW", "CFAR_CPU.csv")),

        ("CPU AVX",        os.path.join(base, "CPU_FMCW", "FFT_AVX.csv"),
                           os.path.join(base, "CPU_FMCW", "CFAR_AVX.csv")),

        ("GPU 2DFFT",      os.path.join(base, "GPU_FMCW", "FFT_2DFFT.csv"),
                           os.path.join(base, "GPU_FMCW", "CFAR_2DFFT.csv")),

        ("GPU ManuelSHM",  os.path.join(base, "GPU_FMCW", "FFT_ManuelSHM.csv"),
                           os.path.join(base, "GPU_FMCW", "CFAR_ManuelSHM.csv")),
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

    # KRİTİK: Tek show() -> tüm figürler aynı anda açılır
    plt.show()


if __name__ == "__main__":
    main()
