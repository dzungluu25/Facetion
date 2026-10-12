<div align="center">

# 🙂 Facetion

**Real-time face recognition from your webcam, set up in 3 commands.**

Runs entirely on your own computer. No cloud, no GPU, no account.

[![CI](https://github.com/dzungluu25/Facetion/actions/workflows/ci.yml/badge.svg)](https://github.com/dzungluu25/Facetion/actions/workflows/ci.yml)
![Python](https://img.shields.io/badge/python-3.9%2B-blue)
![OpenCV](https://img.shields.io/badge/OpenCV-contrib-green)
![Platforms](https://img.shields.io/badge/platform-Windows%20%7C%20macOS%20%7C%20Linux-lightgrey)
[![Stars](https://img.shields.io/github/stars/dzungluu25/Facetion?style=social)](https://github.com/dzungluu25/Facetion/stargazers)

![demo](assets/demo.gif)

</div>

```bash
pip install git+https://github.com/dzungluu25/Facetion.git
facetion capture --name Alice     # take 50 photos of a face
facetion train                    # learn the faces
facetion recognize                # see names on live video
```

Want better accuracy? Add `--backend dnn` to `train` and `recognize` to use deep-learning models.

---

## ✨ Features

- ⚡ **Fast:** recognizes faces in real time on an ordinary laptop CPU
- 🎯 **Two engines:** a classic LBPH model with nothing to download, or deep-learning models (YuNet + SFace) for much higher accuracy
- 🖼️ **Webcam, photos and videos:** recognize faces live, or in an image or video file, and save the labeled result
- 🔒 **Private:** photos and models never leave your computer
- 👥 **Multiple people:** register as many people as you like, and list or remove them with one command
- 🧰 **Simple CLI:** one `facetion` command with clear options and helpful error messages
- 🖥️ **Cross-platform:** tested on Windows, macOS and Linux
- 🐍 **Hackable:** a small, readable codebase that's good for learning computer vision

## 📦 Installation

You need **Python 3.9+** and a **webcam**.

<details>
<summary>Don't have Python or Git yet?</summary>

- **Python:** download it from [python.org/downloads](https://www.python.org/downloads/). On Windows, tick **"Add python.exe to PATH"** during installation.
- **Git:** download it from [git-scm.com/downloads](https://git-scm.com/downloads). On macOS you can run `xcode-select --install` instead.

Check that they're installed with `python --version` and `git --version`.
</details>

**Option A: install the command (recommended)**

```bash
pip install git+https://github.com/dzungluu25/Facetion.git
```

**Option B: clone the source**

```bash
git clone https://github.com/dzungluu25/Facetion.git
cd Facetion
python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install -e .
```

## 🚀 Usage

### 1. Register a person

```bash
facetion capture --name Alice
```

Look at the camera and move your head slightly. Facetion takes 50 photos and closes by itself. Repeat this for each person.

### 2. Train

```bash
facetion train
```

Run this again whenever you add or remove someone.

### 3. Recognize

```bash
facetion recognize
```

Known faces get a green box with a name and a match score. Unknown faces get a red **"Who are you?"** box. Press <kbd>Esc</kbd> to quit.

### Photos and videos

```bash
facetion recognize group-photo.jpg                     # show the result in a window
facetion recognize party.mp4 -o party-labeled.mp4      # save a labeled copy of a video
facetion recognize photo.jpg -o out.jpg --no-show      # save the result without opening a window
```

### Better accuracy with deep learning

```bash
facetion train --backend dnn
facetion recognize --backend dnn
```

The `dnn` backend uses OpenCV's [YuNet](https://github.com/opencv/opencv_zoo/tree/main/models/face_detection_yunet) face detector and [SFace](https://github.com/opencv/opencv_zoo/tree/main/models/face_recognition_sface) face recognizer. The models (about 40 MB) download automatically the first time and are cached in `~/.cache/facetion`.

| | `lbph` (default) | `dnn` |
|---|---|---|
| Accuracy | Good in steady lighting | Much better with different lighting, angles and expressions |
| Speed | Very fast | Real time on most laptops |
| Download | None | ~40 MB, one time |
| Photos needed per person | ~50 | Works with just a few |

### Command reference

| Command | What it does |
|---|---|
| `facetion capture [--name NAME] [--samples N] [--camera N]` | Register a new person |
| `facetion train [--backend lbph\|dnn]` | Train the model on all captured faces |
| `facetion recognize [FILE] [--backend lbph\|dnn] [-o OUT] [--no-show]` | Recognize faces from the webcam, or from an image or video file |
| `facetion recognize … --threshold N` | Match strictness. For `lbph`, lower is stricter (default `51`). For `dnn`, higher is stricter (default `0.363`) |
| `facetion recognize … --camera N` | Use a different camera |
| `facetion users list` | Show registered people and their IDs |
| `facetion users remove ID` | Delete a person and their photos |
| `facetion --data-dir DIR …` | Store data somewhere other than `./data` (or set `FACETION_DATA_DIR`) |
| `facetion --help` | Show all options |

> You can also run `python -m facetion …` instead of `facetion …`.

## 🧠 How it works

```
  capture                  train                    recognize
┌───────────────┐      ┌────────────────┐       ┌──────────────────┐
│ Webcam        │      │ Face photos    │       │ Webcam           │
│ → detect face │ ───▶ │ → LBPH model   │ ────▶ │ → detect face    │
│ → save photos │      │ → trainer.yml  │       │ → predict name   │
└───────────────┘      └────────────────┘       └──────────────────┘
```

| Step | `lbph` backend | `dnn` backend |
|---|---|---|
| Find faces | [Haar cascade](https://docs.opencv.org/4.x/db/d28/tutorial_cascade_classifier.html) | [YuNet](https://github.com/opencv/opencv_zoo/tree/main/models/face_detection_yunet) neural network |
| Identify faces | [LBPH](https://docs.opencv.org/4.x/df/d25/classcv_1_1face_1_1LBPHFaceRecognizer.html) texture histograms, closest distance | [SFace](https://github.com/opencv/opencv_zoo/tree/main/models/face_recognition_sface) 128-number face embeddings, cosine similarity |

Data is stored in `./data` by default:

```
data/
├── names.json      # person ID → name
├── images/         # face photos (user-<id>-<n>.jpg)
├── trainer.yml     # trained lbph model
└── embeddings.npz  # trained dnn model
```

## 🛠️ Troubleshooting

| Problem | Fix |
|---|---|
| `facetion: command not found` | Use `python -m facetion …`, or make sure your virtual environment is active |
| `module 'cv2' has no attribute 'face'` | `pip uninstall -y opencv-python opencv-contrib-python && pip install opencv-contrib-python` |
| `Could not open camera 0` | Close other apps that use the camera, try `--camera 1`, and on macOS allow camera access for your terminal in **System Settings → Privacy & Security → Camera** |
| Wrong name, or "Who are you?" for a known person | Try `--backend dnn`. Otherwise, use good lighting, capture more photos of that person, and run `facetion train` again |
| `No trained dnn model` | Run `facetion train --backend dnn` (each backend is trained separately) |
| Model download fails | Check your internet connection and retry. You can set `FACETION_MODELS_DIR` to choose where models are stored |

## 🗺️ Roadmap

- [x] Recognize faces in image and video files, not just the webcam
- [x] Deep-learning backend (OpenCV YuNet + SFace) for better accuracy
- [ ] Simple desktop GUI
- [ ] Publish to PyPI (`pip install facetion`)
- [ ] Attendance mode: log who was seen and when, to CSV

Have an idea? [Open an issue](https://github.com/dzungluu25/Facetion/issues/new/choose).

## 🤝 Contributing

Contributions are welcome! See [CONTRIBUTING.md](CONTRIBUTING.md) for setup, tests, and guidelines. Items on the roadmap are a good place to start.

## ⭐ Support

If Facetion is useful to you, please **give it a star**. It helps other people find the project.

## 🙏 Acknowledgements

- [OpenCV](https://opencv.org/) for the Haar cascade and LBPH face recognizer
- [OpenCV Zoo](https://github.com/opencv/opencv_zoo) for the YuNet (MIT) and SFace (Apache 2.0) models
