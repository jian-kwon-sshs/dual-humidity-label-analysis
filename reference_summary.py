# reference_summary.py
#
# 기준 패치(white / gray / black)의 ΔE 통계를 정리해서
# - reference_dE_stats.csv  (mean, std, max, median)
# - reference_dE_mean_bar.png (평균 ΔE 막대그래프 + 에러바)
# - reference_dE_max_heatmap.png (max ΔE 히트맵)
# 을 out_folder 안에 저장한다.

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

OUT_DIR = Path("out_folder")
CSV = OUT_DIR / "all_rois_lab_timeseries.csv"   # CORR 버전 보고 싶으면 여기만 바꾸면 됨

print("CSV 읽는 중:", CSV)
df = pd.read_csv(CSV)
df["time_min"] = df["time_s"] / 60.0

containers = sorted(
    {col.split("_white_L")[0] for col in df.columns if col.endswith("_white_L")}
)
patch_roles = ["white", "gray", "black"]
print("containers:", containers)

# ΔE 계산 함수 ------------------------------------------------------
def compute_dE_series(L, a, b):
    L0, a0, b0 = L.iloc[0], a.iloc[0], b.iloc[0]
    return np.sqrt((L - L0) ** 2 + (a - a0) ** 2 + (b - b0) ** 2)


# 1) 통계 표 만들기 -------------------------------------------------
rows = []
for cname in containers:
    for role in patch_roles:
        Lcol = f"{cname}_{role}_L"
        acol = f"{cname}_{role}_a"
        bcol = f"{cname}_{role}_b"
        if Lcol not in df.columns:
            continue

        d = df[[Lcol, acol, bcol]].copy()
        dE = compute_dE_series(d[Lcol], d[acol], d[bcol])

        rows.append(
            {
                "container": cname,
                "patch": role,
                "dE_mean": dE.mean(),
                "dE_std": dE.std(),
                "dE_max": dE.max(),
                "dE_median": dE.median(),
            }
        )

stats_df = pd.DataFrame(rows)
STATS_CSV = OUT_DIR / "reference_dE_stats.csv"
stats_df.to_csv(STATS_CSV, index=False, encoding="utf-8-sig")
print("통계 CSV 저장:", STATS_CSV)
print(stats_df)

# 2) 평균 ΔE 막대 + 에러바 -----------------------------------------
print("평균 ΔE 막대그래프 그리는 중...")

plt.figure(figsize=(7, 4))
for i, role in enumerate(patch_roles):
    sub = stats_df[stats_df["patch"] == role].set_index("container")
    x = np.arange(len(containers))
    means = sub.loc[containers, "dE_mean"]
    stds = sub.loc[containers, "dE_std"]
    offset = (i - 1) * 0.25
    plt.bar(
        x + offset,
        means,
        yerr=stds,
        width=0.25,
        label=role,
        capsize=3,
    )

plt.xticks(np.arange(len(containers)), containers)
plt.ylabel("ΔE_mean ± std")
plt.title("Reference patch stability (mean ΔE)")
plt.grid(axis="y", linestyle="--", alpha=0.5)
plt.legend()
plt.tight_layout()

BAR_PNG = OUT_DIR / "reference_dE_mean_bar.png"
plt.savefig(BAR_PNG, dpi=200)
plt.close()
print("  -> 저장:", BAR_PNG)

# 3) max ΔE 히트맵 --------------------------------------------------
print("max ΔE 히트맵 그리는 중...")

heat_data = np.zeros((len(patch_roles), len(containers)))
for i, role in enumerate(patch_roles):
    for j, cname in enumerate(containers):
        row = stats_df[
            (stats_df["patch"] == role) & (stats_df["container"] == cname)
        ]
        if not row.empty:
            heat_data[i, j] = row["dE_max"].values[0]

plt.figure(figsize=(6, 4))
im = plt.imshow(heat_data, aspect="auto", origin="lower")
plt.xticks(np.arange(len(containers)), containers)
plt.yticks(np.arange(len(patch_roles)), patch_roles)
plt.colorbar(im, label="max ΔE")
plt.title("Reference patch max ΔE (heatmap)")
plt.tight_layout()

HEAT_PNG = OUT_DIR / "reference_dE_max_heatmap.png"
plt.savefig(HEAT_PNG, dpi=200)
plt.close()
print("  -> 저장:", HEAT_PNG)

print("\n기준 패치 안정성 요약 완료 ✅")
