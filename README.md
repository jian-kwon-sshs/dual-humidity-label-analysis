# CIE Lab Colorimetric Analysis Pipeline

Quantitative colorimetric analysis pipeline for dual temperature–humidity responsive labels using CIE Lab color space.

This code accompanies the paper:

> **Quantitative CIE Lab colorimetric analysis of dual temperature-humidity responsive labels for passive environmental sensing**  
> Jian Kwon, Bomin Kim, Hwajeong Cheong  
> Seoul Science High School

---

## Overview

This pipeline extracts CIE Lab color timeseries from video recordings of humidity-responsive labels under controlled relative humidity (RH) conditions (LiCl: 18%, MgCl₂: 30%, NaCl: 45%, water: ~50%). It applies gray-patch-based illumination correction, spike removal, and ΔE analysis to quantify color change as a function of humidity.

---

## Requirements

- Python >= 3.9
- opencv-python
- numpy
- pandas
- matplotlib

Install dependencies:

    pip install opencv-python numpy pandas matplotlib

---

## File Structure

    ├── run_full_pipeline.py              # Run the entire pipeline at once
    ├── video_lab_timeseries_fixedroi.py  # Step 1: Extract Lab timeseries from video
    ├── analyze_from_csv.py               # Step 2: Raw Lab analysis
    ├── report_plots.py                   # Step 3: Generate raw report plots
    ├── analyze_from_csv_renov.py         # Step 4: Illumination correction + corrected plots
    ├── reference_summary.py              # Step 5: Reference patch ΔE statistics
    ├── roi_config.json                   # ROI definitions for each container
    └── out_folder/                       # Output directory (auto-created)

---

## Usage

Run the full pipeline:

    python run_full_pipeline.py

With custom sampling interval (e.g., 30 seconds):

    python run_full_pipeline.py -i 30

---

## Results Summary

| Condition | RH (%) | Final ΔE |
|-----------|--------|----------|
| LiCl      | 18     | 1.00     |
| MgCl₂     | 30     | 1.56     |
| NaCl      | 45     | 6.44     |
| Water     | ~50    | 10.65    |

ΔE increases monotonically with RH, following an exponential pattern (R² = 0.9957).

---

## License

This code is released for academic and research purposes.
