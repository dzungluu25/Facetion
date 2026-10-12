# Facetion

> Real-time face recognition from your webcam, in three short Python scripts, built on OpenCV.

![Python](https://img.shields.io/badge/python-3.8%2B-blue)
![OpenCV](https://img.shields.io/badge/OpenCV-contrib-green)

![demo](assets/demo.gif)

Facetion detects faces with a Haar cascade and identifies them with OpenCV's **LBPH** (Local Binary Patterns Histograms) recognizer. You capture face samples, train a model, and run live recognition. All of it runs locally with no cloud services and no GPU.

---

## Table of Contents

- [Features](#features)
- [How It Works](#how-it-works)
- [Requirements](#requirements)
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Configuration](#configuration)
- [Troubleshooting](#troubleshooting)
- [Known Limitations](#known-limitations)
- [Contributing](#contributing)

---

## Features

- 📸 **Face capture:** collects grayscale face samples from your webcam automatically
- 🧠 **Training:** trains an LBPH model in seconds on a CPU
- 🎥 **Live recognition:** draws a box, the person's name, and a confidence score on each face
- 👥 **Multiple users:** register as many people as you like. IDs are assigned automatically and stored in `names.json`

## How It Works

```
 capture                train                    recognize
┌──────────────┐      ┌───────────────┐        ┌──────────────────┐
│ Webcam →     │      │ data/images/→ │        │ Webcam →         │
│ detect face →│ ───▶ │ LBPH training │ ─────▶ │ detect + predict │
│ data/images/ │      │ →trainer.yml  │        │ → name on screen │
└──────────────┘      └───────────────┘        └──────────────────┘
        └──── data/names.json (ID → name) ──────────────┘
```

## Requirements

- Python 3.8+
- A webcam
- `opencv-contrib-python` (it provides the `cv2.face` module), `numpy`, and `pillow`

## Installation

```bash
# 1. Clone the repository
git clone https://github.com/dzungluu25/Facetion.git
cd Facetion

# 2. (Recommended) Create a virtual environment
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt
```

> ⚠️ Install **only** `opencv-contrib-python`. Having it alongside `opencv-python` can cause `AttributeError: module 'cv2' has no attribute 'face'`. If you get that error, run `pip uninstall opencv-python opencv-contrib-python`, then `pip install opencv-contrib-python`.

## Usage

Run the three steps in order from the project root.

### 1. Capture face samples

```bash
python -m facetion capture
```

- Enter a name when prompted and press <kbd>Enter</kbd>.
- Look at the camera and keep your face centered. Move your head slightly and change your expression to get more varied samples.
- The script saves **50** face images to `data/images/user-<id>-<n>.jpg` and stops on its own. Press <kbd>Esc</kbd> to stop early.
- The ID-to-name mapping is written to `data/names.json`.

Repeat this step for each person you want to recognize.

### 2. Train the model

```bash
python -m facetion train
```

This reads every image in `data/images/`, trains the LBPH recognizer, and saves the model to `data/trainer.yml`.

> Re-run training every time you add a new person.

### 3. Run real-time recognition

```bash
python -m facetion recognize
```

Each recognized face gets a name and a score. Unrecognized faces are labeled **"Who are you?"**. Press <kbd>Esc</kbd> to quit.

## Project Structure

```
Facetion/
├── facetion/
│   ├── __main__.py      # CLI: python -m facetion <capture|train|recognize>
│   ├── config.py        # Paths and tunable settings
│   ├── utils.py         # Shared helpers (camera, detector, names.json)
│   ├── capture.py       # Step 1: capture face images
│   ├── train.py         # Step 2: train LBPH model
│   └── recognize.py     # Step 3: live recognition
├── models/
│   └── haarcascade_frontalface_default.xml  # Pre-trained Haar face detector
├── data/
│   ├── names.json       # ID → name mapping
│   ├── images/          # (generated, git-ignored) face samples
│   └── trainer.yml      # (generated, git-ignored) trained model
├── assets/demo.gif
├── requirements.txt
└── README.md
```

## Configuration

All settings live in [`facetion/config.py`](facetion/config.py):

| Setting | Default | Description |
|---|---|---|
| `SAMPLES_PER_USER` | `50` | Face images captured per person. More samples usually improve accuracy |
| `CAMERA_INDEX` | `0` | Change this if you have several cameras |
| `FRAME_WIDTH` / `FRAME_HEIGHT` | `640×480` | Camera resolution |
| `CONFIDENCE_THRESHOLD` | `51` | Max LBPH distance accepted as a match (lower = stricter) |

`data/names.json` example:

```json
{
    "0": "Joe",
    "1": "Jane"
}
```

## Troubleshooting

| Problem | Fix |
|---|---|
| `module 'cv2' has no attribute 'face'` | Install `opencv-contrib-python` (see [Installation](#installation)) |
| Camera window doesn't open / black frame | Check the camera index, and grant camera permission to your terminal (macOS: *System Settings → Privacy & Security → Camera*) |
| Model not found | Run `train` before `recognize` |
| No samples found during training | Run `capture` first |
| Wrong names shown | Make sure the keys in `data/names.json` match the IDs in `data/images/` file names, then retrain |

## Known Limitations

- Haar cascades work best on frontal, well-lit faces.
- LBPH is a classical algorithm. For higher accuracy, consider deep-learning embeddings (e.g. `face_recognition`, InsightFace).

## Contributing

Contributions are welcome:

1. Fork the repo
2. Create a branch: `git checkout -b feature/my-feature`
3. Commit your changes: `git commit -m "Add my feature"`
4. Push and open a Pull Request

## Acknowledgements

- [OpenCV](https://opencv.org/) for the Haar cascade and LBPH face recognizer
