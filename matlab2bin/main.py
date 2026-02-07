
import numpy as np
import scipy.io
from pathlib import Path
import os

"""
klasor_yolu = "radar_raw_frame"
# Sadece .mat ile biten dosyaların isimlerini al
mat_isimleri = [f for f in os.listdir(klasor_yolu) if f.endswith('.mat')]

for dosya in mat_isimleri:
    matDosyalari = "radar_raw_frame/" + dosya.split('.')[0]
    data = scipy.io.loadmat(matDosyalari)
    adc_data = data['adcData'] # Mevcut: (128, 255, 4, 2)

    # Boyutları senin istediğin sıraya getir: [TX][RX][Chirp][Sample]
    # Eksenler: 3 (TX), 2 (RX), 1 (Chirp), 0 (Sample)
    transposed = adc_data.transpose(3, 2, 1, 0) # Yeni: (2, 4, 255, 128)

    # C++ için I ve Q (Real/Imag) değerlerini ardışık hale getir (Interleaved)
    flattened = np.zeros(transposed.size * 2, dtype=np.float32)
    flattened[0::2] = transposed.real.flatten()
    flattened[1::2] = transposed.imag.flatten()
    binDosyalari = "radar_raw_frameBin/" + dosya.split('.')[0] + ".bin"

    #print(binDosyalari)
    # Bin dosyası olarak kaydet
    flattened.tofile(binDosyalari)
    
"""
""" 

# 255 chirp u 256 ya padding eklenmis hali
### 255 degerine badding ekleyecegim
import scipy.io
import numpy as np

# 1. .mat dosyasını yükle
mat_data = scipy.io.loadmat('radar_raw_frame/000005.mat')
adc_data = mat_data['adcData'] # Orijinal: (128, 255, 4, 2)

# 2. Padded (Sıfır eklenmiş) boş bir dizi oluştur (128, 256, 4, 2)
padded_shape = (128, 256, 4, 2)
padded_adc = np.zeros(padded_shape, dtype=adc_data.dtype)

# 3. Orijinal veriyi yeni dizinin içine yerleştir (ilk 255 chirp dolar, 256. chirp 0 kalır)
padded_adc[:, :255, :, :] = adc_data

# 4. İstediğiniz düzen: [TX][RX][Chirp][Sample]
# Eksenler: (3:TX, 2:RX, 1:Chirp, 0:Sample)
converted_data = padded_adc.transpose(3, 2, 1, 0)

# 5. C++ float complex uyumu için complex64 yap (Real 32bit + Imag 32bit)
converted_data = converted_data.astype(np.complex64)

# 6. Binary olarak kaydet
with open('adcData_padded.bin', 'wb') as f:
    f.write(converted_data.tobytes())

print(f"İşlem Tamam: 255 -> 256 chirp. Yeni boyut: {converted_data.shape}")

"""


klasor_yolu = "radar_raw_frame"
# Sadece .mat ile biten dosyaların isimlerini al
mat_isimleri = [f for f in os.listdir(klasor_yolu) if f.endswith('.mat')]



for dosya in mat_isimleri:
    binName = "../radar_raw_frameBin/" + dosya.split('.')[0]
    matDosyalari = "radar_raw_frame/" + dosya
    data = scipy.io.loadmat(matDosyalari)
    adc_data = data['adcData'] # Mevcut: (128, 255, 4, 2)

    #print(f"İlk Değer (Python [1,0,0,0]): {adc_data[1,0,0,0]}")
    padded_shape = (128, 256, 4, 2)
    padded_adc = np.zeros(padded_shape, dtype=adc_data.dtype)

    # 3. Orijinal veriyi yeni dizinin içine yerleştir (ilk 255 chirp dolar, 256. chirp 0 kalır)
    padded_adc[:, :255, :, :] = adc_data

    # Boyutları senin istediğin sıraya getir: [TX][RX][Chirp][Sample]
    # Eksenler: 3 (RX), 2 (TX), 1 (Chirp), 0 (Sample)
    #converted_data = padded_adc.transpose(3, 2, 1, 0) # Yeni: (4, 2, 255, 128)

    converted_data = np.ascontiguousarray(padded_adc, dtype=np.complex64)


    #print(f"İlk Değer (Python [0,0,0,0]): {converted_data[1,0,0,0]}")


    # 6. Binary olarak kaydet
    with open(binName + '.bin', 'wb') as f:
        f.write(converted_data.tobytes())
    