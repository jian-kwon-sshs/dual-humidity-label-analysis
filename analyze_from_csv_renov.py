# analyze_from_csv_renov.py
#
# 1) out_folder/all_rois_lab_timeseries.csv 를 읽어서
#    "회색(gray) 패치"만 기준으로 조명/밝기 변화를 보정
# 2) 보정된 label Lab 에 대해 스파이크 제거
# 3) ΔE_from_initial, progress_to_waterFinal, t50/t90 계산
# 4) 보정 Lab 기반 그래프들(CORR) 생성
#
# 출력 (모두 out_folder 안):
#   - all_rois_lab_timeseries_CORRECTED.csv
#   - label_only_timeseries_CORR.csv
#   - label_summary_final_CORR.csv
#   - label_speed_t50_t90_CORR.csv
#   - ab_trajectory_subplots_CORR.png
#   - ab_trajectory_combined_CORR.png
#   - a_b_vs_time_CORR.png
#   - dE_vs_time_CORR.png
#   - final_dE_bar_CORR.png
#   - t50_t90_bar_CORR.png

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import math

OUT_DIR = Path("out_folder")
RAW_CSV = OUT_DIR / "all_rois_lab_timeseries.csv"

# ============================================================
# 스파이크 제거 함수들
# ============================================================

def despike_series(series, window=7, z_thresh=5.0):
    """
    중앙값 기반 스파이크 제거.
    - window: rolling median 윈도우 크기 (홀수 권장)
    - z_thresh: robust z-score 기준. 클수록 덜 민감.
    """
    s = series.astype(float)
    med = s.rolling(window=window, center=True, min_periods=1).median()
    diff = s - med
    mad = diff.abs().median()  # median absolute deviation

    if mad == 0 or np.isnan(mad):
        # 거의 변동이 없는 경우: 그대로 반환
        return s

    z = diff.abs() / (1.4826 * mad)
    mask = z > z_thresh

    s_filt = s.copy()
    s_filt[mask] = med[mask]
    return s_filt


def despike_lab_per_container(label_df, window=7, z_thresh=5.0):
    """
    label_df (columns: container, time_s, time_min, L, a, b, ...) 에 대해
    각 container별 L,a,b 시계열의 스파이크를 제거한 DataFrame 반환.
    """
    out = []
    for cname, d in label_df.groupby("container", sort=False):
        d = d.sort_values("time_s").copy()
        for ch in ["L", "a", "b"]:
            d[ch] = despike_series(d[ch], window=window, z_thresh=z_thresh)
        out.append(d)
    return pd.concat(out, ignore_index=True)

# ============================================================
# 1) 회색(gray) 패치로 Lab 보정
# ============================================================

print("raw CSV 읽는 중:", RAW_CSV)
df = pd.read_csv(RAW_CSV)
df["time_min"] = df["time_s"] / 60.0

# container 이름 추출
containers = sorted(
    {col.split("_label_L")[0] for col in df.columns if col.endswith("_label_L")}
)
print("containers:", containers)

# 각 container의 첫 프레임 gray Lab 값을 baseline으로 사용
first_row = df.iloc[0]
baseline_gray = {}
for cname in containers:
    Lcol = f"{cname}_gray_L"
    acol = f"{cname}_gray_a"
    bcol = f"{cname}_gray_b"
    if not all(c in df.columns for c in [Lcol, acol, bcol]):
        raise ValueError(f"{cname} 에 gray 패치 Lab 컬럼이 없습니다.")
    baseline_gray[cname] = np.array(
        [first_row[Lcol], first_row[acol], first_row[bcol]]
    )

# 프레임별 gray 변화량만큼 label에서 빼서 보정
corr_df = df.copy()

for cname in containers:
    Lg = f"{cname}_gray_L"
    ag = f"{cname}_gray_a"
    bg = f"{cname}_gray_b"
    base_L, base_a, base_b = baseline_gray[cname]

    dL = corr_df[Lg] - base_L
    dA = corr_df[ag] - base_a
    dB = corr_df[bg] - base_b

    Llab = f"{cname}_label_L"
    alab = f"{cname}_label_a"
    blab = f"{cname}_label_b"

    if not all(c in df.columns for c in [Llab, alab, blab]):
        raise ValueError(f"{cname} 에 label Lab 컬럼이 없습니다.")

    corr_df[f"{cname}_label_L_corr"] = corr_df[Llab] - dL
    corr_df[f"{cname}_label_a_corr"] = corr_df[alab] - dA
    corr_df[f"{cname}_label_b_corr"] = corr_df[blab] - dB

CORR_ALL_CSV = OUT_DIR / "all_rois_lab_timeseries_CORRECTED.csv"
corr_df.to_csv(CORR_ALL_CSV, index=False, encoding="utf-8-sig")
print("보정 전체 CSV 저장:", CORR_ALL_CSV)

# ============================================================
# 2) 보정된 label Lab → 스파이크 제거 → ΔE, progress 계산
# ============================================================

rows = []
for cname in containers:
    Lc = f"{cname}_label_L_corr"
    ac = f"{cname}_label_a_corr"
    bc = f"{cname}_label_b_corr"

    sub = corr_df[["time_s", "time_min", Lc, ac, bc]].copy()
    sub.rename(columns={Lc: "L", ac: "a", bc: "b"}, inplace=True)
    sub["container"] = cname
    rows.append(sub)

label_corr = pd.concat(rows, ignore_index=True)

print("스파이크 제거 수행(보정 Lab)...")
label_corr = despike_lab_per_container(label_corr, window=7, z_thresh=5.0)

# ΔE_from_initial 계산
def add_deltaE_from_initial(d):
    d = d.sort_values("time_s").copy()
    L0, a0, b0 = d.iloc[0][["L", "a", "b"]]
    d["dE_from_initial"] = np.sqrt(
        (d["L"] - L0) ** 2 +
        (d["a"] - a0) ** 2 +
        (d["b"] - b0) ** 2
    )
    return d

label_corr = (
    label_corr.groupby("container", group_keys=False)
    .apply(add_deltaE_from_initial)
)

# water 기준 컨테이너 선택
water_candidates = [c for c in containers if "water" in c.lower()]
water_name = water_candidates[0] if water_candidates else containers[-1]
print("water 기준 컨테이너:", water_name)

water_final_dE = (
    label_corr[label_corr["container"] == water_name]["dE_from_initial"].max()
)

label_corr["progress_to_waterFinal"] = (
    label_corr["dE_from_initial"] / max(water_final_dE, 1e-6)
)

# CSV 저장
LABEL_CORR_CSV = OUT_DIR / "label_only_timeseries_CORR.csv"
label_corr.to_csv(LABEL_CORR_CSV, index=False, encoding="utf-8-sig")
print("보정 label_only_timeseries 저장:", LABEL_CORR_CSV)

# 최종 요약표
summary_rows = []
for cname in containers:
    d = label_corr[label_corr["container"] == cname].sort_values("time_s")
    last = d.iloc[-1]
    row = {
        "container": cname,
        "final_time_min": last["time_min"],
        "final_L": last["L"],
        "final_a": last["a"],
        "final_b": last["b"],
        "final_dE_from_initial": last["dE_from_initial"],
        "final_progress_to_waterFinal": last["progress_to_waterFinal"],
    }
    summary_rows.append(row)

summary_corr = pd.DataFrame(summary_rows)
SUMMARY_CORR_CSV = OUT_DIR / "label_summary_final_CORR.csv"
summary_corr.to_csv(SUMMARY_CORR_CSV, index=False, encoding="utf-8-sig")
print("보정 label_summary_final 저장:", SUMMARY_CORR_CSV)
print(summary_corr)

# t50 / t90
def find_time_for_progress(d, target):
    d = d.sort_values("time_s")
    after = d[d["progress_to_waterFinal"] >= target]
    if after.empty:
        return None
    return after.iloc[0]["time_min"]

speed_rows = []
for cname in containers:
    d = label_corr[label_corr["container"] == cname]
    t50 = find_time_for_progress(d, 0.5)
    t90 = find_time_for_progress(d, 0.9)
    speed_rows.append({"container": cname, "t50_min": t50, "t90_min": t90})

speed_corr = pd.DataFrame(speed_rows)
SPEED_CORR_CSV = OUT_DIR / "label_speed_t50_t90_CORR.csv"
speed_corr.to_csv(SPEED_CORR_CSV, index=False, encoding="utf-8-sig")
print("보정 속도표(t50, t90) 저장:", SPEED_CORR_CSV)
print(speed_corr)

# ============================================================
# 3) 그래프들 (CORR 버전)
# ============================================================

def downsample_df(d, max_points=200):
    if len(d) <= max_points:
        return d
    idx = np.linspace(0, len(d) - 1, max_points).astype(int)
    return d.iloc[idx]

# 1) a-b 궤적 (per container)
print("a-b 궤적 subplot (보정) 만드는 중...")

n = len(containers)
rows_n = math.ceil(n / 2)
cols_n = 2

fig, axes = plt.subplots(rows_n, cols_n, figsize=(8, 4 * rows_n), squeeze=False)
axes = axes.ravel()

for i, cname in enumerate(containers):
    ax = axes[i]
    d = label_corr[label_corr["container"] == cname].sort_values("time_s")
    d_ds = downsample_df(d, max_points=150)

    sc = ax.scatter(d_ds["a"], d_ds["b"], c=d_ds["time_min"], s=15)
    ax.plot(d_ds["a"], d_ds["b"], linewidth=0.5)
    ax.invert_xaxis()
    ax.set_title(cname)
    ax.set_xlabel("a* (corrected)")
    ax.set_ylabel("b* (corrected)")
    ax.grid(True)
    cb = fig.colorbar(sc, ax=ax)
    cb.set_label("time (min)")

for i in range(len(containers), len(axes)):
    fig.delaxes(axes[i])

fig.suptitle(
    "Corrected label color trajectory in a-b plane (per container)",
    fontsize=14,
)
plt.tight_layout(rect=[0, 0, 1, 0.96])
out_path = OUT_DIR / "ab_trajectory_subplots_CORR.png"
plt.savefig(out_path, dpi=200)
plt.close()
print("  -> 저장:", out_path)

# 2) a-b overlay
print("a-b 궤적 overlay (보정) 만드는 중...")

plt.figure(figsize=(7, 6))
for cname in containers:
    d = label_corr[label_corr["container"] == cname].sort_values("time_s")
    d_ds = downsample_df(d, max_points=150)
    plt.plot(d_ds["a"], d_ds["b"], "-o", markersize=3, label=cname, alpha=0.7)
    plt.text(d_ds["a"].iloc[0], d_ds["b"].iloc[0], f"{cname}_start", fontsize=8)
    plt.text(d_ds["a"].iloc[-1], d_ds["b"].iloc[-1], f"{cname}_end", fontsize=8)

plt.xlabel("a* (corrected)")
plt.ylabel("b* (corrected)")
plt.title("Corrected label color trajectory (overlay)")
plt.gca().invert_xaxis()
plt.legend()
plt.grid(True)
plt.tight_layout()
out_path = OUT_DIR / "ab_trajectory_combined_CORR.png"
plt.savefig(out_path, dpi=200)
plt.close()
print("  -> 저장:", out_path)

# 3) a*(t), b*(t)
print("a*(t), b*(t) (보정) 그래프 만드는 중...")

plt.figure(figsize=(7, 6))

ax1 = plt.subplot(2, 1, 1)
for cname in containers:
    d = label_corr[label_corr["container"] == cname].sort_values("time_s")
    ax1.plot(d["time_min"], d["a"], label=cname)
ax1.set_ylabel("a* (corrected)")
ax1.set_title("a* vs time (corrected)")
ax1.grid(True)
ax1.legend()

ax2 = plt.subplot(2, 1, 2, sharex=ax1)
for cname in containers:
    d = label_corr[label_corr["container"] == cname].sort_values("time_s")
    ax2.plot(d["time_min"], d["b"], label=cname)
ax2.set_xlabel("time (min)")
ax2.set_ylabel("b* (corrected)")
ax2.set_title("b* vs time (corrected)")
ax2.grid(True)

plt.tight_layout()
out_path = OUT_DIR / "a_b_vs_time_CORR.png"
plt.savefig(out_path, dpi=200)
plt.close()
print("  -> 저장:", out_path)

# 4) dE vs time
print("ΔE_from_initial vs time (보정) 그래프 만드는 중...")

plt.figure(figsize=(7, 5))
for cname in containers:
    d = label_corr[label_corr["container"] == cname].sort_values("time_s")
    plt.plot(d["time_min"], d["dE_from_initial"], "-o", markersize=3, label=cname)

plt.xlabel("time (min)")
plt.ylabel("ΔE_from_initial (corrected)")
plt.title("Corrected color change (ΔE) vs time")
plt.grid(True)
plt.legend()
plt.tight_layout()
out_path = OUT_DIR / "dE_vs_time_CORR.png"
plt.savefig(out_path, dpi=200)
plt.close()
print("  -> 저장:", out_path)

# 5) 최종 ΔE 막대
print("최종 ΔE 막대 (보정) 그래프 만드는 중...")

plt.figure(figsize=(6, 4))
x = np.arange(len(summary_corr["container"]))
plt.bar(x, summary_corr["final_dE_from_initial"])
plt.xticks(x, summary_corr["container"])
plt.ylabel("final ΔE_from_initial (corrected)")
plt.title("Final color difference (ΔE, corrected)")
plt.grid(axis="y", linestyle="--", alpha=0.5)
plt.tight_layout()
out_path = OUT_DIR / "final_dE_bar_CORR.png"
plt.savefig(out_path, dpi=200)
plt.close()
print("  -> 저장:", out_path)

# 6) t50 / t90 막대
print("t50, t90 막대 (보정) 그래프 만드는 중...")

plt.figure(figsize=(6, 4))
sub = speed_corr.set_index("container").loc[containers]
x = np.arange(len(sub.index))
width = 0.35
plt.bar(x - width / 2, sub["t50_min"], width, label="t50")
plt.bar(x + width / 2, sub["t90_min"], width, label="t90")

plt.xticks(x, sub.index)
plt.ylabel("time (min)")
plt.title("Response time (t50, t90, corrected)")
plt.legend()
plt.grid(axis="y", linestyle="--", alpha=0.5)
plt.tight_layout()
out_path = OUT_DIR / "t50_t90_bar_CORR.png"
plt.savefig(out_path, dpi=200)
plt.close()
print("  -> 저장:", out_path)

print("\n✅ 회색 패치 기준 보정 + 스파이크 제거 완료.")
