import os
import subprocess
import re
import matplotlib.pyplot as plt
import time

test_configurations = [
    (256, 128),
    (512, 256),
    (1024, 512),
    (1024, 1024),
    (2048, 512),
    (2048, 1024),
    (2048, 2048)
]


COMPILE_CMD = "cd .. && nvcc -arch=sm_86 -std=c++17 -O3 mainLast.cu --options-file compile2.txt"
EXE_NAME = "FMCW_Algo2.exe" 


HEADER_FILE = "../defines.h"

def update_defines(chirps, samples):
    with open(HEADER_FILE, "r", encoding="utf-8") as f:
        content = f.read()
    
    content = re.sub(r"const int NUM_CHIRPS = \d+;", f"const int NUM_CHIRPS = {chirps};", content)
    content = re.sub(r"const int NUM_SAMPLES = \d+;", f"const int NUM_SAMPLES = {samples};", content)
    content = re.sub(r"const int REF_R = \d+;", f"const int REF_R = {chirps/32};", content)
    content = re.sub(r"const int REF_C = \d+;", f"const int REF_C = {chirps/32};", content)
    content = re.sub(r"const int GUARD_R = \d+;", f"const int GUARD_R = {chirps/128};", content)
    content = re.sub(r"const int GUARD_C = \d+;", f"const int GUARD_C = {chirps/128};", content)
    
    with open(HEADER_FILE, "w", encoding="utf-8") as f:
        f.write(content)
    print(f"-> Ayarlar güncellendi: {chirps} x {samples}")
    
    print(f"-> Python Ayarlar güncellendi: {chirps} x {samples}")

def parse_output(output_text):
    
    data = {}
    data2 = {}
    dataCfar = {} 
    
    lines = output_text.split('\n')
    for line in lines:
        if "Manuel FFT" in line:
            t = re.search(r"TOTAL time[:\s]+(\d+\.\d+)", line, re.IGNORECASE)
            c = re.search(r"FMCW time[:\s]+(\d+\.\d+)", line, re.IGNORECASE)
            cfar = re.search(r"CFAR time[:\s]+(\d+\.\d+)", line, re.IGNORECASE)

            if t and c: 
                data['shared_mem_total'] = float(t.group(1))
                data2['shared_mem_fmcw'] = float(c.group(1))
                dataCfar['shared_mem_cfar'] = float(cfar.group(1))

        if "2D_FFT" in line:
            t = re.search(r"TOTAL time[:\s]+(\d+\.\d+)", line, re.IGNORECASE)
            c = re.search(r"FMCW time[:\s]+(\d+\.\d+)", line, re.IGNORECASE)
            cfar = re.search(r"CFAR time[:\s]+(\d+\.\d+)", line, re.IGNORECASE)
            if t and c: 
                data['2DFFT_total'] = float(t.group(1))
                data2['2DFFT_fmcw'] = float(c.group(1))
                dataCfar['2DFFT_cfar'] = float(cfar.group(1))

        if "Recurisive FFT" in line:
            fmcw = re.search(r"FMCW time[:\s]+(\d+\.\d+)", line, re.IGNORECASE)           
            cfar = re.search(r"CFAR time[:\s]+(\d+\.\d+)", line, re.IGNORECASE)           
            total = re.search(r"TOTAL time[:\s]+(\d+\.\d+)", line, re.IGNORECASE)  
            if fmcw and cfar and total:
                data['CPU_total'] = float(total.group(1))
                data2['CPU_fmcw'] = float(fmcw.group(1))
                dataCfar['CPU_cfar'] = float(cfar.group(1))


        if "AVX FFT" in line:
            total = re.search(r"TOTAL time[:\s]+(\d+\.\d+)", line, re.IGNORECASE)  
            fmcw = re.search(r"FMCW time[:\s]+(\d+\.\d+)", line, re.IGNORECASE)           
            cfar = re.search(r"CFAR time[:\s]+(\d+\.\d+)", line, re.IGNORECASE)           
            if fmcw and cfar and total:
                data['AVX_total'] = float(total.group(1))
                data2['AVX_fmcw'] = float(fmcw.group(1))
                dataCfar['AVX_cfar'] = float(cfar.group(1))


    

    return data, data2, dataCfar 

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
            
            totaltimes, computetimes, cfartimes = parse_output(program_output)
            
            if totaltimes and computetimes:
                results.append({
                    "label": f"{chirps}x{samples}",
                    "points": chirps * samples,
                    "totaltimes": totaltimes,
                    "computetimes" :computetimes,
                    "cfartimes": cfartimes
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
        ([r['totaltimes'].get('CPU_total', 0) for r in results], 'CPU OpenMP FFT', 'm', 'v', '--'),
        ([r['totaltimes'].get('AVX_total', 0) for r in results], 'CPU AVX', 'black', '+', '--'),
        ([r['totaltimes'].get('shared_mem_total', 0) for r in results], 'GPU Shared Mem', 'green', 'o', '-'),
        ([r['totaltimes'].get('2DFFT_total', 0) for r in results], 'GPU 2D FFT', 'purple', '*', '-') 
    ]

    y_compute_sets = [
        ([r['computetimes'].get('CPU_fmcw', 0) for r in results], 'CPU OpenMP FFT', 'blue', '^', ':'),
        ([r['computetimes'].get('AVX_fmcw', 0) for r in results], 'CPU AVX', 'red', 's', '-.'),
        ([r['computetimes'].get('shared_mem_fmcw', 0) for r in results], 'GPU Shared Mem', 'green', 'o', '-'),
        ([r['computetimes'].get('2DFFT_fmcw', 0) for r in results], 'GPU 2D FFT', 'purple', '*', '-') # Marker değişti (*)
    ]
    y_cfar_compute_sets = [
        ([r['cfartimes'].get('CPU_cfar', 0) for r in results], 'CPU OpenMP FFT', 'blue', '^', ':'),
        ([r['cfartimes'].get('AVX_cfar', 0) for r in results], 'CPU AVX', 'red', 's', '-.'),
        ([r['cfartimes'].get('shared_mem_cfar', 0) for r in results], 'GPU Shared Mem', 'green', 'o', '-'),
        ([r['cfartimes'].get('2DFFT_cfar', 0) for r in results], 'GPU 2D FFT', 'purple', '*', '-') # Marker değişti (*)
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
    
    plt.title("FMCW-CFAR Radar Processing: Total Execution Time")
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
    
    plt.figure(figsize=(14, 8))

    for data, label, color, marker, style in y_cfar_compute_sets:
        plt.plot(labels, data, label=label, color=color, marker=marker, linestyle=style)
        
        for i, val in enumerate(data):
            if val > 0:
                plt.annotate(f"{val:.2f}", 
                             xy=(i, val), 
                             xytext=(0, 5), 
                             textcoords="offset points",
                             ha='center', va='bottom', 
                             fontsize=8, fontweight='bold', color=color)

    plt.title("CFAR Processing: Compute Time Only (Log Scale)")
    plt.xlabel("Configuration (Chirps x Samples)")
    plt.ylabel("Time (ms)")
    plt.yscale('log')
    plt.legend()
    plt.grid(True, alpha=0.3, which="both", ls="-") 
    plt.tight_layout()
    plt.savefig("benchmark_cfar_compute_time.png")
    print("Grafik cfar Kaydedildi: benchmark_cfar_compute_time.png")
    

    plt.show()

if __name__ == "__main__":

    all_data = run_tests()
    plot_benchmark(all_data)
    