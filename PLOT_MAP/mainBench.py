import os
import subprocess
import re
import matplotlib.pyplot as plt
import time

test_configurations = [
    (32, 16),
    (64, 32),
    (128, 64),
    (256, 128),
    (512, 256),
    (1024, 512),
    (1024, 1024),
    (2048, 512),
    (2048, 1024),
    (2048, 2048),
]


COMPILE_CMD = "cd .. && nvcc -arch=sm_86 -std=c++17 -O3 mainLast.cu --options-file compile2.txt"
EXE_NAME = "FMCW_Algo2.exe" 


HEADER_FILE = "../defines.h"
PYTHON_FILE = "main.py"

def update_defines(chirps, samples):
    
    with open(HEADER_FILE, "r") as f:
        content = f.read()
    
    content = re.sub(r"const int NUM_CHIRPS = \d+;", f"const int NUM_CHIRPS = {chirps};", content)
    content = re.sub(r"const int NUM_SAMPLES = \d+;", f"const int NUM_SAMPLES = {samples};", content)
    
    with open(HEADER_FILE, "w") as f:
        f.write(content)
    print(f"-> Ayarlar güncellendi: {chirps} x {samples}")

    with open(PYTHON_FILE, "r") as f:
        content = f.read()
    
    content = re.sub(r"NUM_CHIRPS = \d+", f"NUM_CHIRPS = {chirps}", content)
    content = re.sub(r"NUM_SAMPLES = \d+", f"NUM_SAMPLES = {samples}", content)

    with open(PYTHON_FILE, "w") as f:
        f.write(content)
    print(f"-> Python Ayarlar güncellendi: {chirps} x {samples}")

def parse_output(output_text):
    
    data = {}
    data2 = {}
        
    lines = output_text.split('\n')
    for line in lines:
        if "Shared yok" in line:
            t = re.search(r"total time[:\s]+(\d+\.\d+)", line, re.IGNORECASE)
            c = re.search(r"Compute time[:\s]+(\d+\.\d+)", line, re.IGNORECASE)

            if t and c: 
                data['shared_yok_total'] = float(t.group(1))
                data2['shared_yok_compute'] = float(c.group(1))
            
        if "Shared Mem" in line:
            t = re.search(r"total time[:\s]+(\d+\.\d+)", line, re.IGNORECASE)
            c = re.search(r"Compute time[:\s]+(\d+\.\d+)", line, re.IGNORECASE)

            if t and c: 
                data['shared_mem_total'] = float(t.group(1))
                data2['shared_mem_compute'] = float(c.group(1))

        if "Shared Stream" in line:
            t = re.search(r"total time[:\s]+(\d+\.\d+)", line, re.IGNORECASE)
            c = re.search(r"Compute time[:\s]+(\d+\.\d+)", line, re.IGNORECASE)

            if t and c: 
                data['shared_stream_total'] = float(t.group(1))
                data2['shared_stream_compute'] = float(c.group(1))

        if "1DFFT" in line:
            t = re.search(r"total time[:\s]+(\d+\.\d+)", line, re.IGNORECASE)
            c = re.search(r"Compute time[:\s]+(\d+\.\d+)", line, re.IGNORECASE)

            if t and c: 
                data['1DFFT_total'] = float(t.group(1))
                data2['1DFFT_compute'] = float(c.group(1))

        if "2D_FFT" in line:
            t = re.search(r"total time[:\s]+(\d+\.\d+)", line, re.IGNORECASE)
            c = re.search(r"Compute time[:\s]+(\d+\.\d+)", line, re.IGNORECASE)

            if t and c: 
                data['2DFFT_total'] = float(t.group(1))
                data2['2DFFT_compute'] = float(c.group(1))

        if "Recursive FFT" in line:
            normal = re.search(r"FFT time[:\s]+(\d+\.\d+)", line, re.IGNORECASE)           
            openMP = re.search(r"OpenMP time[:\s]+(\d+\.\d+)", line, re.IGNORECASE)           
            AVX = re.search(r"AVX Total time[:\s]+(\d+\.\d+)", line, re.IGNORECASE)  
            if normal and openMP and AVX:
                data['CPU_Normal'] = float(normal.group(1))
                data['CPU_openMP'] = float(openMP.group(1))
                data['CPU_AVX'] = float(AVX.group(1))


    return data, data2 

def run_tests():
    results = []
    parent_dir = ".."

    for chirps, samples in test_configurations:
        print(f"\n==========================================")
        print(f"TEST BAŞLIYOR: Chirps={chirps}, Samples={samples}")
        print(f"==========================================")
        
        update_defines(chirps, samples)
        print("Derleniyor... (Biraz sürebilir)")
        try:
            subprocess.run(COMPILE_CMD, shell=True, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        except subprocess.CalledProcessError as e:
            print("!!! DERLEME HATASI OLUŞTU !!!")
            print(e.stderr.decode('cp1254', errors='ignore')) # Türkçe karakter sorunu olmasın diye
            continue 

        print("Program çalıştırılıyor...")
        try:
            if os.name == 'nt':
                cmd = f".\\{EXE_NAME}"
            else:
                cmd = f"./{EXE_NAME}"
                
            process = subprocess.run(cmd, shell=True, check=True, 
                                     stdout=subprocess.PIPE, text=True,
                                     cwd=parent_dir)
            program_output = process.stdout
            
            totaltimes, computetimes = parse_output(program_output)
            
            if totaltimes and computetimes:
                results.append({
                    "label": f"{chirps}x{samples}",
                    "points": chirps * samples,
                    "totaltimes": totaltimes,
                    "computetimes" :computetimes
                })
            else:
                print("UYARI: Çıktıdan süre okunamadı. Program çıktısı:")
                print(program_output[:300])

        except Exception as e:
            print(f"Çalıştırma hatası: {e}")
        
    return results

def plot_benchmark(results):
    if not results:
        print("Grafik çizilecek veri yok.")
        return

    labels = [r['label'] for r in results]
    
    y_total_sets = [
        ([r['totaltimes'].get('CPU_Normal', 0) for r in results], 'CPU Recursive FFT', 'm', 'v', '--'),
        ([r['totaltimes'].get('CPU_openMP', 0) for r in results], 'CPU Recursive FFT OpenMP', 'pink', 'D', '--'),
        ([r['totaltimes'].get('CPU_AVX', 0) for r in results], 'CPU AVX', 'black', '+', '--'),
        ([r['totaltimes'].get('shared_yok_total', 0) for r in results], 'Shared Yok', 'orange', 'x', '--'),
        ([r['totaltimes'].get('shared_mem_total', 0) for r in results], 'Shared Mem', 'green', 'o', '-'),
        ([r['totaltimes'].get('shared_stream_total', 0) for r in results], 'Shared Stream', 'blue', '^', ':'),
        ([r['totaltimes'].get('1DFFT_total', 0) for r in results], '1D FFT', 'red', 's', '-.'),
        ([r['totaltimes'].get('2DFFT_total', 0) for r in results], '2D FFT', 'purple', '*', '-') 
    ]

    y_compute_sets = [
        ([r['computetimes'].get('shared_yok_compute', 0) for r in results], 'Shared Yok', 'orange', 'x', '--'),
        ([r['computetimes'].get('shared_mem_compute', 0) for r in results], 'Shared Mem', 'green', 'o', '-'),
        ([r['computetimes'].get('shared_stream_compute', 0) for r in results], 'Shared Stream', 'blue', '^', ':'),
        ([r['computetimes'].get('1DFFT_compute', 0) for r in results], '1D FFT', 'red', 's', '-.'),
        ([r['computetimes'].get('2DFFT_compute', 0) for r in results], '2D FFT', 'purple', '*', '-') # Marker değişti (*)
    ]

    plt.figure(figsize=(14, 8))

    for data, label, color, marker, style in y_total_sets:
        plt.plot(labels, data, label=label, color=color, marker=marker, linestyle=style)
        for i, val in enumerate(data):
            if val > 0:
                plt.annotate(f"{val:.2f}", 
                             xy=(i, val), 
                             xytext=(0, 5), 
                             textcoords="offset points",
                             ha='center', va='bottom', 
                             fontsize=8, fontweight='bold', color=color)
    
    plt.title("FMCW Radar Processing: Total Execution Time")
    plt.xlabel("Configuration (Chirps x Samples)")
    plt.ylabel("Time (ms)")
    plt.yscale('log') 
    plt.legend()
    plt.grid(True, alpha=0.3, which="both", ls="-") 
    plt.tight_layout()
    plt.savefig("benchmark_total_time.png")

    plt.figure(figsize=(14, 8))

    for data, label, color, marker, style in y_compute_sets:
        plt.plot(labels, data, label=label, color=color, marker=marker, linestyle=style)
        
        for i, val in enumerate(data):
            if val > 0:
                plt.annotate(f"{val:.2f}", 
                             xy=(i, val), 
                             xytext=(0, 5), 
                             textcoords="offset points",
                             ha='center', va='bottom', 
                             fontsize=8, fontweight='bold', color=color)

    plt.title("FMCW Radar Processing: Compute Time Only (Log Scale)")
    plt.xlabel("Configuration (Chirps x Samples)")
    plt.ylabel("Time (ms)")
    plt.yscale('log')
    plt.legend()
    plt.grid(True, alpha=0.3, which="both", ls="-") 
    plt.tight_layout()
    plt.savefig("benchmark_compute_time.png")
    print("Grafik 2 Kaydedildi: benchmark_compute_time.png")
    
    plt.show()

if __name__ == "__main__":

    all_data = run_tests()
    plot_benchmark(all_data)
    