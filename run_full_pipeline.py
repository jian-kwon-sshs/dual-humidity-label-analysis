# run_full_pipeline.py
#
# 전체 Lab 분석 파이프라인 실행:
#   1) video_lab_timeseries_fixedroi.py  (ROI 고정, Lab 시계열 추출)
#   2) analyze_from_csv.py               (raw Lab → CSV + 기본 a-b 궤적)
#   3) report_plots.py                   (raw Lab 보고서용 그래프들)
#   4) analyze_from_csv_renov.py         (기준 패치로 보정 + 보정 그래프들)
#   5) check_reference_stability.py      (기준 패치 ΔE vs time)
#   6) reference_summary.py              (기준 패치 ΔE 통계/히트맵)
#
# 사용 예:
#   python run_full_pipeline.py              # 기본: 10초 간격
#   python run_full_pipeline.py -i 30        # 30초 간격
#
# 같은 폴더에 있어야 하는 파일:
#   - labels_video_finalversion.mp4
#   - roi_config.json
#   - video_lab_timeseries_fixedroi.py
#   - analyze_from_csv.py
#   - report_plots.py
#   - analyze_from_csv_renov.py
#   - check_reference_stability.py
#   - reference_summary.py

import argparse
import subprocess
import sys


def run(cmd: list[str]):
    print("\n==================================================")
    print("실행:", " ".join(cmd))
    print("==================================================\n")
    subprocess.run(cmd, check=True)


def main():
    parser = argparse.ArgumentParser(
        description="전체 Lab 분석 파이프라인 실행 스크립트"
    )
    parser.add_argument(
        "-i", "--interval",
        type=float,
        default=10.0,
        help="영상 샘플링 간격 (초, 기본 10초)",
    )
    parser.add_argument(
        "-v", "--video",
        default="labels_video_finalversion.mp4",
        help="입력 비디오 파일 이름 (기본: labels_video_finalversion.mp4)",
    )
    args = parser.parse_args()

    # 1) 영상 → all_rois_lab_timeseries.csv
    run([sys.executable, "video_lab_timeseries_fixedroi.py",
         "-v", args.video,
         "-i", str(args.interval)])

    # 2) raw Lab 분석
    run([sys.executable, "analyze_from_csv.py"])

    # 3) raw 보고서용 그래프
    run([sys.executable, "report_plots.py"])

    # 4) 기준 패치 보정 + 보정 그래프
    run([sys.executable, "analyze_from_csv_renov.py"])

    # 5) 기준 패치 안정성 (ΔE vs time)
    run([sys.executable, "check_reference_stability.py"])

    # 6) 기준 패치 ΔE 통계/히트맵
    run([sys.executable, "reference_summary.py"])

    print("\n✅ 전체 파이프라인 완료.")
    print("   - 원본 Lab CSV/그래프: out_folder/*.csv, out_folder/*.png")
    print("   - 보정 Lab CSV/그래프: out_folder/*_CORR.csv, out_folder/*_CORR.png")


if __name__ == "__main__":
    main()
