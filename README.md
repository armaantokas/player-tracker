# player-tracker

#Player Re-Identification & Tracking

A computer vision pipeline that detects, tracks, and re-identifies football players in a video — even if they leave and re-enter the frame — using a combination of YOLOv5 for detection, custom SORT tracking, and feature-based re-identification.

---

## Features

- **Player Detection** using a custom YOLOv5 model trained on football data.
- **Object Tracking** via a modified SORT algorithm.
- **Re-Identification** of players who exit and re-enter the frame.
- Real-time **bounding boxes** with consistent player IDs.
- Optional video output saving and display.

---

## Setup Instructions

### 1. Clone the Repository
```bash
git clone https://github.com/armaantokas/player-tracker.git
cd player-tracker
```
### 2.Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Download YOLOv5 Weights
The model (`best.pt`) is not included in the repo due to size limits.

To download it:
```bash
python download_model.py
```

If that does not work here is the link to download:- "https://drive.google.com/file/d/1kTXyF9O4gCGHIVsM9HfRr7X3eopc4zj4/view?usp=sharing"

### 4. Run Main.py

---

### Author
Armaan Tokas

"https://www.github.com/armaantokas"
