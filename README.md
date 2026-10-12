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

---

## ✨ Features

- ⚡ **Fast:** recognizes faces in real time on an ordinary laptop CPU
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

Known faces are labeled with a name and a match score. Unknown faces are labeled **"Who are you?"**. Press <kbd>Esc</kbd> to quit.

### Command reference

| Command | What it does |
|---|---|
| `facetion capture [--name NAME] [--samples N] [--camera N]` | Register a new person |
| `facetion train` | Train the model on all captured faces |
| `facetion recognize [--threshold N] [--camera N]` | Live recognition. A lower threshold is stricter (default `51`) |
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

1. **Detection:** a [Haar cascade](https://docs.opencv.org/4.x/db/d28/tutorial_cascade_classifier.html) finds faces in each frame.
2. **Recognition:** OpenCV's [LBPH](https://docs.opencv.org/4.x/df/d25/classcv_1_1face_1_1LBPHFaceRecognizer.html) recognizer compares each face with the people you registered and returns the closest match and its distance.

Data is stored in `./data` by default:

```
data/
├── names.json      # person ID → name
├── images/         # face photos (user-<id>-<n>.jpg)
└── trainer.yml     # trained model
```

## 🛠️ Troubleshooting

| Problem | Fix |
|---|---|
| `facetion: command not found` | Use `python -m facetion …`, or make sure your virtual environment is active |
| `module 'cv2' has no attribute 'face'` | `pip uninstall -y opencv-python opencv-contrib-python && pip install opencv-contrib-python` |
| `Could not open camera 0` | Close other apps that use the camera, try `--camera 1`, and on macOS allow camera access for your terminal in **System Settings → Privacy & Security → Camera** |
| Wrong name, or "Who are you?" for a known person | Use good lighting, capture more photos of that person, and run `facetion train` again. You can also try `--threshold 60` |

## 🗺️ Roadmap

- [ ] Recognize faces in image and video files, not just the webcam
- [ ] Deep-learning backend (OpenCV YuNet + SFace) for better accuracy
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
