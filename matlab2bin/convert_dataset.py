import numpy as np
import scipy.io
import os

NUM_SAMPLES = 128
NUM_CHIRPS  = 256

hanning_range   = np.hanning(NUM_SAMPLES).astype(np.float32)
hanning_doppler = np.hanning(NUM_CHIRPS).astype(np.float32)
window_2d = np.outer(hanning_doppler, hanning_range)   # (256, 128)

DATASET_DIR = "2019_04_09_bms1000/radar_raw_frame"
OUTPUT_DIR  = "2019_04_09_bms1000/radar_bin"

mat_files = sorted([f for f in os.listdir(DATASET_DIR) if f.endswith('.mat')])
total = len(mat_files)

print(f"Toplam {total} frame donusturuluyor...")

for i, dosya in enumerate(mat_files):
    src = os.path.join(DATASET_DIR, dosya)
    dst = os.path.join(OUTPUT_DIR, dosya.replace('.mat', '.bin'))

    if os.path.exists(dst):
        continue  # zaten donusturulduyse atla

    data     = scipy.io.loadmat(src)
    adc_data = data['adcData']               # (128, 255, 4, 2)

    padded = np.zeros((128, 256, 4, 2), dtype=adc_data.dtype)
    padded[:, :255, :, :] = adc_data        # 255 -> 256 chirp zero-pad

    # [Sample, Chirp, RX, TX] -> [TX, RX, Chirp, Sample]
    converted = padded.transpose(3, 2, 1, 0)
    converted = np.ascontiguousarray(converted, dtype=np.complex64)
    converted *= window_2d[np.newaxis, np.newaxis, :, :]

    with open(dst, 'wb') as f:
        f.write(converted.tobytes())

    if (i + 1) % 100 == 0 or (i + 1) == total:
        print(f"  [{i+1}/{total}] {dosya} -> {os.path.basename(dst)}")

print("Donusum tamamlandi.")
