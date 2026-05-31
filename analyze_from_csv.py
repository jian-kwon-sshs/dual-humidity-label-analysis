# analyze_from_csv.py
#
# out_folder/all_rois_lab_timeseries.csv 를 읽어서
# 1) 각 container의 label Lab 시계열을 정리
# 2) ΔE_from_initial, progress_to_waterFinal 계산
# 3) 아래 파일 저장:
#    - label_only_timeseries.csv
#    - label_summary_final.csv
#    - label_speed_t50_t90.csv
#    - label_ab_trajectory.png (간단한 a-b 궤적)

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

OUT_DIR = Path("out_folder")
ALL_CSV = OUT_DIR / "all_rois_lab_timeseries.csv"

print("CSV 읽는 중:", ALL_CSV)
df = pd.read_csv(ALL_CSV)

# time_s -> time_min
df["time_min"] = df["time_s"] / 60.0

# container 이름 자동 추출: "{container}_label_L" 패턴
containers = sorted(
    {col[:-len("_label_L")] for col in df.columns if col.endswith("_label_L")}
)
print("containers:", containers)

# 1) label_only_timeseries 만들기 ---------------------------------
rows = []
for cname in containers:
    Lcol = f"{cname}_label_L"
    acol = f"{cname}_label_a"
    bcol = f"{cname}_label_b"

    sub = df[["time_s", "time_min", Lcol, acol, bcol]].copy()
    sub.rename(
        columns={
            Lcol: "L",
            acol: "a",
            bcol: "b",
        },
        inplace=True,
    )
    sub["container"] = cname
    rows.append(sub)

label_df = pd.concat(rows, ignore_index=True)

# 2) ΔE_from_initial 계산 -----------------------------------------
def add_deltaE_from_initial(d):
    d = d.sort_values("time_s").copy()
    L0, a0, b0 = d.iloc[0][["L", "a", "b"]]
    d["dE_from_initial"] = np.sqrt(
        (d["L"] - L0) ** 2 +
        (d["a"] - a0) ** 2 +
        (d["b"] - b0) ** 2
    )
    return d


label_df = (
    label_df.groupby("container", group_keys=False)
    .apply(add_deltaE_from_initial)
)

# water 기준 컨테이너 선택
water_candidates = [c for c in containers if "water" in c.lower()]
if water_candidates:
    water_name = water_candidates[0]
else:
    water_name = containers[-1]
print("water 기준 컨테이너:", water_name)

water_final_dE = (
    label_df[label_df["container"] == water_name]["dE_from_initial"].max()
)

label_df["progress_to_waterFinal"] = (
    label_df["dE_from_initial"] / max(water_final_dE, 1e-6)
)

# 3) label_only_timeseries.csv 저장 -------------------------------
label_csv = OUT_DIR / "label_only_timeseries.csv"
label_df.to_csv(label_csv, index=False, encoding="utf-8-sig")
print("label_only_timeseries 저장:", label_csv)

# 4) 최종 요약표 --------------------------------------------------
summary_rows = []
for cname in containers:
    d = label_df[label_df["container"] == cname].sort_values("time_s")
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

summary = pd.DataFrame(summary_rows)
summary_path = OUT_DIR / "label_summary_final.csv"
summary.to_csv(summary_path, index=False, encoding="utf-8-sig")
print("label_summary_final 저장:", summary_path)
print(summary)

# 5) t50 / t90 계산 -----------------------------------------------

def find_time_for_progress(d, target):
    d = d.sort_values("time_s")
    after = d[d["progress_to_waterFinal"] >= target]
    if after.empty:
        return None
    return after.iloc[0]["time_min"]


speed_rows = []
for cname in containers:
    d = label_df[label_df["container"] == cname]
    t50 = find_time_for_progress(d, 0.5)
    t90 = find_time_for_progress(d, 0.9)
    speed_rows.append(
        {
            "container": cname,
            "t50_min": t50,
            "t90_min": t90,
        }
    )

speed = pd.DataFrame(speed_rows)
speed_path = OUT_DIR / "label_speed_t50_t90.csv"
speed.to_csv(speed_path, index=False, encoding="utf-8-sig")
print("속도표(t50, t90) 저장:", speed_path)
print(speed)

# 6) 간단 a-b 궤적 그래프 ----------------------------------------

plt.figure(figsize=(6, 6))
for cname in containers:
    d = label_df[label_df["container"] == cname].sort_values("time_s")
    a_vals = d["a"].values
    b_vals = d["b"].values
    plt.plot(a_vals, b_vals, "-o", label=cname, markersize=3)
    plt.text(a_vals[0], b_vals[0], cname + "_start", fontsize=8)
    plt.text(a_vals[-1], b_vals[-1], cname + "_end", fontsize=8)

plt.xlabel("a*")
plt.ylabel("b*")
plt.title("Label color trajectory in a-b plane")
plt.grid(True)
plt.legend()
plt.gca().invert_xaxis()
plt.tight_layout()
ab_png = OUT_DIR / "label_ab_trajectory.png"
plt.savefig(ab_png, dpi=200)
plt.close()
print("a-b 궤적 그래프 저장:", ab_png)
