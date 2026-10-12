<div align="center">

# 🙂 Facetion

**Recognize faces and control your computer with hand gestures, using just your webcam.**

Runs entirely on your own computer. No cloud, no GPU, no account, no special hardware.

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

**🖐️ Control your computer with your hand:**

```bash
pip install "facetion[control] @ git+https://github.com/dzungluu25/Facetion.git"
facetion control                  # point to move, pinch to click
facetion control --owner Alice    # only works while Alice is in front of the camera
```

---

## ✨ Features

- 🖐️ **Hand-gesture control:** move the mouse, click, drag, right-click, scroll and play/pause music with your hand
- 🔐 **Face-locked control:** only the people you choose can control the computer with gestures
- ⚡ **Fast and light:** 60–130 fps hand tracking and real-time face recognition on an ordinary laptop CPU, with about 50 MB of models in total
- 🎯 **Two engines:** a classic LBPH model with nothing to download, or deep-learning models (YuNet + SFace) for much higher accuracy
- 🖼️ **Webcam, photos and videos:** recognize faces live, or in an image or video file, and save the labeled result
- 🔒 **Private:** photos and models never leave your computer
- 👥 **Multiple people:** register as many people as you like, and list or remove them with one command
- 🧰 **Simple CLI:** one `facetion` command with clear options and helpful error messages
- 🖥️ **Runs anywhere OpenCV runs:** Windows, macOS (Intel and Apple Silicon), Linux, even a Raspberry Pi. Only OpenCV is needed, not MediaPipe
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

**Add hand control:** this also installs [pynput](https://github.com/moses-palmer/pynput) for mouse and keyboard control.

```bash
pip install "facetion[control] @ git+https://github.com/dzungluu25/Facetion.git"
```

**Option B: clone the source**

```bash
git clone https://github.com/dzungluu25/Facetion.git
cd Facetion
python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install -e ".[control]"
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

### 🖐️ Hand-gesture control

```bash
facetion control
```

A window shows your hand and the gesture it detected. Your webcam image is mirrored, so it works like a mirror.

| Gesture | Action |
|---|---|
| ☝️ **Point** (index finger up) | Move the mouse |
| 🤏 **Pinch** thumb + index | Click. Hold the pinch and move to drag |
| 🤌 **Pinch** thumb + middle finger | Right-click |
| ✌️ **Peace sign** | Scroll: move your hand up or down |
| 👍 **Thumbs up** (hold ½ s) | Play / pause media |
| ✊ **Fist** or 🖐️ **open palm** | Pause, so you can move your hand without moving the mouse |

Press <kbd>P</kbd> in the window to pause or resume, and <kbd>Esc</kbd> to quit.

**Face lock:** register yourself first (`facetion capture --name Alice` and then `facetion train`), then run:

```bash
facetion control --owner Alice
```

Gestures only work while Alice's face is in view. If someone else steps in front of the camera, control locks.

**Tips**
- Try `facetion control --dry-run` first. It shows the detected gestures without moving the mouse.
- Is the cursor shaky? Lower `--smoothing` (e.g. `0.2`). Does it lag? Raise it (e.g. `0.6`).
- Can't reach the screen edges? Raise `--margin` (e.g. `0.25`).
- **macOS:** give your terminal permission under **System Settings → Privacy & Security → Accessibility** (to control the mouse) and **Camera**.
- **Linux:** mouse control needs an X11 session. On Wayland, log in with "Ubuntu on Xorg".

### Command reference

| Command | What it does |
|---|---|
| `facetion capture [--name NAME] [--samples N] [--camera N]` | Register a new person |
| `facetion train [--backend lbph\|dnn]` | Train the model on all captured faces |
| `facetion recognize [FILE] [--backend lbph\|dnn] [-o OUT] [--no-show]` | Recognize faces from the webcam, or from an image or video file |
| `facetion recognize … --threshold N` | Match strictness. For `lbph`, lower is stricter (default `51`). For `dnn`, higher is stricter (default `0.363`) |
| `facetion recognize … --camera N` | Use a different camera |
| `facetion control [--owner NAME] [--dry-run]` | Control the mouse with hand gestures |
| `facetion control … --smoothing N --margin N` | Tune cursor smoothness and how far you need to move your hand |
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

**Hand control** uses MediaPipe's palm detector and 21-point hand-landmark models, [converted to ONNX by OpenCV Zoo](https://github.com/opencv/opencv_zoo/tree/main/models/handpose_estimation_mediapipe), so only OpenCV is needed. After the first detection, the hand is tracked from frame to frame, which skips palm detection and roughly doubles the speed. Gestures are worked out from the finger positions and must hold steady for 3 frames before anything happens, so a single misread frame never clicks.

```
webcam → palm detector → 21 hand points → gesture → mouse / keyboard
              ↑ skipped while tracking        ↑ optional face lock
```

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
| Hand control: mouse doesn't move (macOS) | Allow your terminal under **System Settings → Privacy & Security → Accessibility**, then restart the terminal |
| Hand control: `needs extra packages` | `pip install "facetion[control] @ git+https://github.com/dzungluu25/Facetion.git"` |
| Hand control: cursor jumps around | Use good lighting, keep one hand in view, and lower `--smoothing` |
| Model download fails | Check your internet connection and retry. You can set `FACETION_MODELS_DIR` to choose where models are stored |

## 🗺️ Roadmap

- [x] Recognize faces in image and video files, not just the webcam
- [x] Deep-learning backend (OpenCV YuNet + SFace) for better accuracy
- [x] Hand-gesture mouse control with face lock
- [ ] Custom gestures: map your own gestures to keyboard shortcuts
- [ ] Two-hand gestures (zoom, rotate)
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
- [OpenCV Zoo](https://github.com/opencv/opencv_zoo) for the YuNet (MIT) and SFace (Apache 2.0) face models, and the ONNX versions of MediaPipe's palm and hand models (Apache 2.0)
- [MediaPipe](https://github.com/google-ai-edge/mediapipe) for the original hand-tracking models
- [pynput](https://github.com/moses-palmer/pynput) for mouse and keyboard control
