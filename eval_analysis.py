"""
eval_analysis.py  --  Tespit kalitesi analizi (düzeltilmiş Pd)
Pd = "en az 1 tespit ile yakalanan GT hedef sayısı" / "toplam GT hedef sayısı"
FP = "hiçbir GT hedefine uymayan tespit sayısı"
"""

import os, math
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ─── Yollar ─────────────────────────────────────────────────────────────────
DETECTIONS_CSV = "eval_detections.csv"
LABELS_DIR     = "matlab2bin/2019_04_09_bms1000/text_labels"
RANGE_TOL      = 3.0   # m  (ana tolerans)

# ─── 1. Tespitleri yükle ────────────────────────────────────────────────────
det = pd.read_csv(DETECTIONS_CSV, dtype={"frame_id": str})
det["range_m"] = det["range_m"].astype(float)
det["power"]   = det["power"].astype(float)

# ─── 2. Etiketleri yükle (frame_id → [range_m]) ─────────────────────────────
labels = {}
for lf in sorted(os.listdir(LABELS_DIR)):
    if not lf.endswith(".csv"): continue
    frame_6 = f'{int(lf.split(".")[0]):06d}'
    rows = []
    with open(os.path.join(LABELS_DIR, lf)) as f:
        for line in f:
            parts = line.strip().split(",")
            if len(parts) < 4: continue
            x, y = float(parts[2]), float(parts[3])
            rows.append(math.sqrt(x*x + y*y))
    if rows:
        labels[frame_6] = rows

total_gt = sum(len(v) for v in labels.values())
print(f"Toplam GT hedef : {total_gt}  ({len(labels)} frame)")

# ─── 3. TP/FP hesapla ─────────────────────────────────────────────────────
# Her GT için "yakalandı mı?" bayrağı
# Her tespit için "bu bir GT'ye uyan tespit mi?" bayrağı (FP analizi için)

def compute_stats(det_df, tol):
    """Returns: (tp_gt, total_gt, fp_count, tp_det_mask)"""
    tp_gt     = 0
    fp_mask   = pd.Series(True, index=det_df.index)  # başlangıçta hepsi FP say

    by_frame = {fid: grp for fid, grp in det_df.groupby("frame_id")}

    for fid, gt_ranges in labels.items():
        grp = by_frame.get(fid)
        for gt_r in gt_ranges:
            if grp is not None:
                hit = (grp["range_m"] - gt_r).abs() <= tol
                if hit.any():
                    tp_gt += 1
                    fp_mask.loc[grp[hit].index] = False  # GT'ye uyanlar FP değil

    fp_count = fp_mask.sum()
    return tp_gt, total_gt, fp_count, fp_mask

tp_gt, _, fp_count, fp_mask = compute_stats(det, RANGE_TOL)
fp = det[fp_mask]
tp = det[~fp_mask]

print(f"\n{'─'*55}")
print(f"  Toplam tespit       : {len(det)}")
print(f"  GT'ye uyan tespit   : {len(tp)}  (TP-det, leakage dahil)")
print(f"  FP tespit           : {fp_count}")
print(f"  FP/frame            : {fp_count / det['frame_id'].nunique():.1f}")
print(f"  GT yakalanan        : {tp_gt} / {total_gt}")
print(f"  Pd (±{RANGE_TOL}m)         : {tp_gt/total_gt*100:.1f}%")
print(f"{'─'*55}\n")

# ─── 4. Pd vs dB threshold eğrisi ───────────────────────────────────────────
dbs   = np.arange(5, 41, 1)
pds   = []
fpfs  = []
n_frames = det["frame_id"].nunique()

print("dB eşiği taraması...")
for db in dbs:
    factor = 10**(db / 10.0)
    keep = pd.Series(True, index=det.index)
    for fid, group in det.groupby("frame_id"):
        for grp_name in ["MOVING", "STATIC"]:
            gidx = group[group["group"] == grp_name].index
            if len(gidx) == 0: continue
            max_p = group.loc[gidx, "power"].max()
            min_p = max_p / factor
            keep[gidx[group.loc[gidx, "power"] < min_p]] = False
    kept = det[keep]
    tp_k, _, fp_k, _ = compute_stats(kept, RANGE_TOL)
    pds.append(tp_k / total_gt * 100)
    fpfs.append(fp_k / n_frames)

# ─── 5. Grafik ───────────────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 3, figsize=(16, 9))
fig.suptitle("False Alarm Analizi  (filter açık, dB_thr=20)", fontsize=13)

ax = axes[0, 0]
ax.hist(fp["range_m"], bins=50, color="salmon", edgecolor="k", linewidth=0.4)
ax.set_xlabel("Mesafe (m)"); ax.set_ylabel("FP sayısı"); ax.set_title("FP – Range Dağılımı")

ax = axes[0, 1]
ax.hist(fp["vel_ms"], bins=60, color="salmon", edgecolor="k", linewidth=0.4)
ax.set_xlabel("Hız (m/s)"); ax.set_title("FP – Velocity Dağılımı")

ax = axes[0, 2]
grp_cnt = fp["group"].value_counts()
ax.bar(grp_cnt.index, grp_cnt.values, color=["steelblue","orange"])
ax.set_title("FP – Group (STATIC/MOVING)"); ax.set_ylabel("FP sayısı")

ax = axes[1, 0]
ax.hist(np.log10(tp["power"].clip(1)), bins=40, alpha=0.6, label="GT uyan", color="green", edgecolor="k", linewidth=0.3)
ax.hist(np.log10(fp["power"].clip(1)), bins=40, alpha=0.6, label="FP",      color="red",   edgecolor="k", linewidth=0.3)
ax.set_xlabel("log10(Power)"); ax.set_title("Power: GT-uyan vs FP"); ax.legend()

ax = axes[1, 1]
fp_per_frame = fp.groupby("frame_id").size()
ax.hist(fp_per_frame, bins=30, color="coral", edgecolor="k", linewidth=0.4)
ax.set_xlabel("FP / frame"); ax.set_ylabel("Frame sayısı"); ax.set_title("Frame başına FP")

ax = axes[1, 2]
ax2 = ax.twinx()
ax.plot(dbs, pds,  "b-o", ms=3, label="Pd (%)")
ax2.plot(dbs, fpfs, "r-s", ms=3, label="FP/frame")
ax.axvline(20, color="gray", ls="--", lw=1, label="mevcut (20dB)")
ax.set_xlabel("dB Eşiği"); ax.set_ylabel("Pd (%)", color="b")
ax2.set_ylabel("FP/frame", color="r")
ax.set_title("Pd  &  FP/frame  vs  dB Eşiği")
lines1, labs1 = ax.get_legend_handles_labels()
lines2, labs2 = ax2.get_legend_handles_labels()
ax.legend(lines1+lines2, labs1+labs2, loc="center right", fontsize=8)


plt.tight_layout()
plt.savefig("PLOT_MAP/eval_analysis.png", dpi=130)
print("Grafik → PLOT_MAP/eval_analysis.png")

# ─── 6. Özet tablo ───────────────────────────────────────────────────────────
print(f"\n{'dB':>4} | {'Pd(%)':>7} | {'FP/fr':>7}")
print("─"*25)
for db, pd_v, fpf in zip(dbs, pds, fpfs):
    marker = " ◄" if db == 20 else ""
    print(f"{db:>4} | {pd_v:>7.1f} | {fpf:>7.1f}{marker}")

best_db, best_fpf, best_pd = None, 1e9, 0
for db, pd_v, fpf in zip(dbs, pds, fpfs):
    if pd_v >= 95.0 and fpf < best_fpf:
        best_fpf = fpf; best_db = db; best_pd = pd_v

if best_db:
    print(f"\nPd≥95% koşulunda min FP/frame  →  dB={best_db}, Pd={best_pd:.1f}%, FP/frame={best_fpf:.1f}")
plt.show()    
