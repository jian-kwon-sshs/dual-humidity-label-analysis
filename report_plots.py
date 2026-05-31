# report_plots.py
#
# analyze_from_csv.py 결과(out_folder/*.csv)를 이용해
# 보고서/발표용 그래프들을 한 번에 그리는 스크립트.
#
# 생성되는 파일 (out_folder 안):
#   - ab_trajectory_subplots.png
#   - ab_trajectory_combined.png
#   - a_b_vs_time.png
#   - dE_vs_time.png
#   - final_dE_bar.png
#   - t50_t90_bar.png

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import math

OUT_DIR = Path("out_folder")
LABEL_CSV = OUT_DIR / "label_only_timeseries.csv"
SUMMARY_CSV = OUT_DIR / "label_summary_final.csv"
SPEED_CSV = OUT_DIR / "label_speed_t50_t90.csv"

print("label_only_timeseries 읽는 중:", LABEL_CSV)
label_df = pd.read_csv(LABEL_CSV)
label_df["time_min"] = label_df["time_s"] / 60.0

summary = pd.read_csv(SUMMARY_CSV)
speed = pd.read_csv(SPEED_CSV)

containers = list(label_df["container"].unique())
print("containers:", containers)


def downsample_df(d, max_points=200):
    """포인트가 너무 많을 때, 대략 max_points 개 정도로 줄이는 함수."""
    if len(d) <= max_points:
        return d
    idx = np.linspace(0, len(d) - 1, max_points).astype(int)
    return d.iloc[idx]


# ===== 1. a-b 궤적 (container별 subplot) ==========================
print("1) a-b 궤적 subplot 그림 만드는 중...")

n = len(containers)
rows = math.ceil(n / 2)
cols = 2

fig, axes = plt.subplots(rows, cols, figsize=(8, 4 * rows), squeeze=False)
axes = axes.ravel()

for i, cname in enumerate(containers):
    ax = axes[i]
    d = label_df[label_df["container"] == cname].sort_values("time_s")
    d_ds = downsample_df(d, max_points=150)

    sc = ax.scatter(d_ds["a"], d_ds["b"], c=d_ds["time_min"], s=15)
    ax.plot(d_ds["a"], d_ds["b"], linewidth=0.5)
    ax.invert_xaxis()
    ax.set_title(cname)
    ax.set_xlabel("a*")
    ax.set_ylabel("b*")
    ax.grid(True)

    cb = fig.colorbar(sc, ax=ax)
    cb.set_label("time (min)")

for i in range(len(containers), len(axes)):
    fig.delaxes(axes[i])

fig.suptitle("Label color trajectory in a-b plane (per container)", fontsize=14)
plt.tight_layout(rect=[0, 0, 1, 0.96])
out_path = OUT_DIR / "ab_trajectory_subplots.png"
plt.savefig(out_path, dpi=200)
plt.close()
print("  -> 저장:", out_path)

# ===== 2. a-b 궤적 overlay =======================================
print("2) a-b 궤적 overlay 그림 만드는 중...")

plt.figure(figsize=(7, 6))
for cname in containers:
    d = label_df[label_df["container"] == cname].sort_values("time_s")
    d_ds = downsample_df(d, max_points=150)
    plt.plot(d_ds["a"], d_ds["b"], "-o", markersize=3, label=cname, alpha=0.7)
    plt.text(d_ds["a"].iloc[0], d_ds["b"].iloc[0], f"{cname}_start", fontsize=8)
    plt.text(d_ds["a"].iloc[-1], d_ds["b"].iloc[-1], f"{cname}_end", fontsize=8)

plt.xlabel("a*")
plt.ylabel("b*")
plt.title("Label color trajectory (overlay)")
plt.gca().invert_xaxis()
plt.legend()
plt.grid(True)
plt.tight_layout()
out_path = OUT_DIR / "ab_trajectory_combined.png"
plt.savefig(out_path, dpi=200)
plt.close()
print("  -> 저장:", out_path)

# ===== 3. a*(t), b*(t) ============================================
print("3) a*(t), b*(t) 시간 그래프 만드는 중...")

plt.figure(figsize=(7, 6))

ax1 = plt.subplot(2, 1, 1)
for cname in containers:
    d = label_df[label_df["container"] == cname].sort_values("time_s")
    ax1.plot(d["time_min"], d["a"], label=cname)
ax1.set_ylabel("a*")
ax1.set_title("a* vs time")
ax1.grid(True)
ax1.legend()

ax2 = plt.subplot(2, 1, 2, sharex=ax1)
for cname in containers:
    d = label_df[label_df["container"] == cname].sort_values("time_s")
    ax2.plot(d["time_min"], d["b"], label=cname)
ax2.set_xlabel("time (min)")
ax2.set_ylabel("b*")
ax2.set_title("b* vs time")
ax2.grid(True)

plt.tight_layout()
out_path = OUT_DIR / "a_b_vs_time.png"
plt.savefig(out_path, dpi=200)
plt.close()
print("  -> 저장:", out_path)

# ===== 4. dE_from_initial vs time ================================
print("4) ΔE_from_initial vs time 그래프 만드는 중...")

plt.figure(figsize=(7, 5))
for cname in containers:
    d = label_df[label_df["container"] == cname].sort_values("time_s")
    plt.plot(d["time_min"], d["dE_from_initial"], "-o", markersize=3, label=cname)

plt.xlabel("time (min)")
plt.ylabel("ΔE_from_initial")
plt.title("Color change (ΔE) vs time")
plt.grid(True)
plt.legend()
plt.tight_layout()
out_path = OUT_DIR / "dE_vs_time.png"
plt.savefig(out_path, dpi=200)
plt.close()
print("  -> 저장:", out_path)

# ===== 5. 최종 ΔE 막대 ===========================================
print("5) 최종 ΔE 막대 그래프 만드는 중...")

plt.figure(figsize=(6, 4))
x = np.arange(len(summary["container"]))
plt.bar(x, summary["final_dE_from_initial"])
plt.xticks(x, summary["container"])
plt.ylabel("final ΔE_from_initial")
plt.title("Final color difference (ΔE)")
plt.grid(axis="y", linestyle="--", alpha=0.5)
plt.tight_layout()
out_path = OUT_DIR / "final_dE_bar.png"
plt.savefig(out_path, dpi=200)
plt.close()
print("  -> 저장:", out_path)

# ===== 6. t50 / t90 막대 =========================================
print("6) t50, t90 막대 그래프 만드는 중...")

plt.figure(figsize=(6, 4))
sub = speed.set_index("container").loc[containers]
x = np.arange(len(sub.index))
width = 0.35
plt.bar(x - width / 2, sub["t50_min"], width, label="t50")
plt.bar(x + width / 2, sub["t90_min"], width, label="t90")

plt.xticks(x, sub.index)
plt.ylabel("time (min)")
plt.title("Response time (t50, t90)")
plt.legend()
plt.grid(axis="y", linestyle="--", alpha=0.5)
plt.tight_layout()
out_path = OUT_DIR / "t50_t90_bar.png"
plt.savefig(out_path, dpi=200)
plt.close()
print("  -> 저장:", out_path)

print("\n모든 raw 그래프 생성 완료 ✅")
