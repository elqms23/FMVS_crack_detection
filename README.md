# CCTV Crack Inspection Pipeline

Heuristic-based crack detection pipeline with automatic ROI tracking.

## Features
- Automatic ROI tracking using segmentation (UNet)
- Heuristic crack detection (black-hat + contour)
- Time-range based inspection
- Crack visualization & saving

## Project Structure

```text
cctv_insp/
├─ src/
│  ├─ main.py            # Entry point
│  ├─ pipeline.py        # Main inspection pipeline
│  ├─ config.py          # Argument parsing & configuration
│  ├─ time_utils.py      # Time parsing utilities (H:M:S, ranges)
│  ├─ utils.py           # Common helper functions
│  │
│  ├─ ROI/
│  │  ├─ autoROI.py      # Automatic ROI tracking using segmentation
│  │  └─ unet.py         # UNet model definition for ROI segmentation
│  │
│  └─ CrackDetect/
│     └─ heuristic.py    # Heuristic crack detection (black-hat + contours)
│
├─ requirements.txt      # Python dependencies
├─ .gitignore            # Git ignore rules (models, videos, outputs)
└─ README.md             # Project documentation
