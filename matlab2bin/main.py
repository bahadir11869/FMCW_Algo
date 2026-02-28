import numpy as np
import scipy.io
from pathlib import Path
import os

NUM_SAMPLES = 128
NUM_CHIRPS  = 256

# 2D Hanning Window'u bir kez oluştur: (Chirp=256, Sample=128)
# np.hanning uç noktaları 0 yaptığı için np.hanning(N) kullanıyoruz
hanning_range   = np.hanning(NUM_SAMPLES).astype(np.float32)   # (128,)
hanning_doppler = np.hanning(NUM_CHIRPS).astype(np.float32)    # (256,)
window_2d = np.outer(hanning_doppler, hanning_range)            # (256, 128)

klasor_yolu = "radarDeneme"
mat_isimleri = [f for f in os.listdir(klasor_yolu) if f.endswith('.mat')]

for dosya in mat_isimleri:
    binName      = "radar_raw_frameBinDeneme/" + dosya.split('.')[0]
    matDosyalari = "radarDeneme/" + dosya

    data     = scipy.io.loadmat(matDosyalari)
    adc_data = data['adcData']  # Orijinal: (128, 255, 4, 2)

    # 255 chirp → 256 chirp (zero padding)
    padded_adc = np.zeros((128, 256, 4, 2), dtype=adc_data.dtype)
    padded_adc[:, :255, :, :] = adc_data

    # [Sample, Chirp, RX, TX] → [TX, RX, Chirp, Sample] = (2, 4, 256, 128)
    converted_data = padded_adc.transpose(3, 2, 1, 0)

    # complex64'e çevir
    converted_data = np.ascontiguousarray(converted_data, dtype=np.complex64)

    # 2D Hanning Window uygula
    # converted_data shape: (TX=2, RX=4, Chirp=256, Sample=128)
    # window_2d shape: (256, 128) → broadcast için (1, 1, 256, 128)
    converted_data *= window_2d[np.newaxis, np.newaxis, :, :]

    # Binary olarak kaydet
    with open(binName + '.bin', 'wb') as f:
        f.write(converted_data.tobytes())

    print(f"[OK] {dosya} → {binName}.bin | shape: {converted_data.shape}")