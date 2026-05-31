# video_lab_timeseries_fixedroi.py
#
# roi_config.json 에 있는 ROI만 사용해서
# labels_video_finalversion.mp4 에서 Lab 시계열을 뽑아
# out_folder/all_rois_lab_timeseries.csv 로 저장.
#
# 사용 예:
#   python video_lab_timeseries_fixedroi.py
#   python video_lab_timeseries_fixedroi.py -i 30
#
# ROI 파일 형식 (roi_config.json):
# {
#   "container_names": ["LiCl", "MgCl2", "NaCl", "water"],
#   "rois": {
#     "LiCl": { "label": [x,y,w,h], "white": [...], "gray": [...], "black": [...] },
#     "MgCl2": { ... },
#     ...
#   }
# }

import cv2
import numpy as np
import json
import argparse
from pathlib import Path

ROI_CONFIG_PATH = "roi_config.json"
OUT_DIR = Path("out_folder")


def bgr_to_opencv_lab(bgr: np.ndarray) -> np.ndarray:
    """BGR(0-255) -> OpenCV Lab(0-255, 128-neutral)."""
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    return lab.astype(np.float32)


def opencv_lab_to_cie_lab(lab_cv: np.ndarray) -> np.ndarray:
    """
    OpenCV Lab(0-255, 128-neutral) -> approximate CIE Lab:

      L* in [0,100], a*, b* around [-128,127]
    """
    lab_cv = lab_cv.astype(np.float32)
    L_cv = lab_cv[..., 0]
    a_cv = lab_cv[..., 1]
    b_cv = lab_cv[..., 2]

    L_star = L_cv * (100.0 / 255.0)
    a_star = a_cv - 128.0
    b_star = b_cv - 128.0

    return np.stack([L_star, a_star, b_star], axis=-1)


def mean_cie_lab_of_roi(frame_bgr: np.ndarray, roi):
    """
    frame_bgr: HxWx3 BGR uint8
    roi: (x, y, w, h)
    returns: (L*, a*, b*) mean over ROI
    """
    x, y, w, h = roi
    crop = frame_bgr[y:y + h, x:x + w]
    if crop.size == 0:
        raise ValueError(f"Empty ROI region for roi={roi}")
    lab_cv = bgr_to_opencv_lab(crop)
    lab_cie = opencv_lab_to_cie_lab(lab_cv)
    mean_lab = lab_cie.reshape(-1, 3).mean(axis=0)
    return mean_lab  # (L*, a*, b*)


def load_rois_from_config(path: str = ROI_CONFIG_PATH):
    """roi_config.json 에서 container_names 와 rois 를 읽어온다."""
    cfg_path = Path(path)
    if not cfg_path.exists():
        raise FileNotFoundError(f"ROI 설정 파일을 찾을 수 없습니다: {cfg_path}")

    with cfg_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    container_names = data.get("container_names", [])
    rois = data.get("rois", {})

    if not container_names or not rois:
        raise ValueError("roi_config.json 내용이 비어 있거나 형식이 잘못되었습니다.")

    for cname in container_names:
        if cname not in rois:
            raise ValueError(f"roi_config.json에 '{cname}'의 ROI 정보가 없습니다.")
        for role in ("label", "white", "gray", "black"):
            if role not in rois[cname]:
                raise ValueError(f"'{cname}'에 '{role}' ROI가 없습니다.")

    print("ROI 설정 로드 완료.")
    print("  containers:", container_names)
    return container_names, rois


def analyze_video(
    video_path: str,
    sample_interval_seconds: float = 10.0,
):
    """
    - video_path 영상을 열고
    - roi_config.json 에서 ROI를 읽어온 뒤
    - sample_interval_seconds 마다 프레임을 샘플링해서
    - 각 ROI의 mean CIE Lab 을 계산하여
      out_folder/all_rois_lab_timeseries.csv 에 저장
    """
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = OUT_DIR / "all_rois_lab_timeseries.csv"

    # ROI 로드
    container_names, container_rois = load_rois_from_config(ROI_CONFIG_PATH)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"비디오를 열 수 없습니다: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 30.0  # fallback
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # CSV 헤더 작성
    cols = ["frame_index", "time_s"]
    for cname in container_names:
        for role in ["label", "white", "gray", "black"]:
            prefix = f"{cname}_{role}"
            cols += [f"{prefix}_L", f"{prefix}_a", f"{prefix}_b"]

    with csv_path.open("w", encoding="utf-8") as f:
        f.write(",".join(cols) + "\n")

    # 샘플링 설정
    step_frames = max(int(sample_interval_seconds * fps), 1)
    print(f"FPS: {fps:.3f}, total_frames: {total_frames}")
    print(f"샘플링 간격: {sample_interval_seconds:.3f} s → {step_frames} 프레임마다 추출")

    print("\n=== 프레임 샘플링 시작 ===")
    frame_index = 0
    while True:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        ret, frame = cap.read()
        if not ret:
            break

        time_s = frame_index / fps
        row_vals = [frame_index, f"{time_s:.3f}"]

        # 각 컨테이너 × 역할에 대해 Lab 계산
        for cname in container_names:
            rois = container_rois[cname]
            for role in ["label", "white", "gray", "black"]:
                roi = tuple(rois[role])
                L, a, b = mean_cie_lab_of_roi(frame, roi)
                row_vals += [f"{L:.4f}", f"{a:.4f}", f"{b:.4f}"]

        with csv_path.open("a", encoding="utf-8") as f:
            f.write(",".join(map(str, row_vals)) + "\n")

        if frame_index % (step_frames * 10) == 0:
            print(f"  frame {frame_index}/{total_frames} (t={time_s:.1f} s)")

        frame_index += step_frames

    cap.release()
    print(f"\n완료: {csv_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Video Lab timeseries extractor (ROI를 roi_config.json에서 고정 사용)"
    )
    parser.add_argument(
        "-v", "--video",
        default="labels_video_finalversion.mp4",
        help="입력 비디오 경로 (기본: labels_video_finalversion.mp4)",
    )
    parser.add_argument(
        "-i", "--interval",
        type=float,
        default=10.0,
        help="샘플링 간격 (초 단위, 기본 10초)",
    )
    args = parser.parse_args()

    analyze_video(
        video_path=args.video,
        sample_interval_seconds=args.interval,
    )


if __name__ == "__main__":
    main()
