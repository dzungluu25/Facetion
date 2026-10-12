# Facetion

> Teach your computer to recognize faces through your webcam.

![Python](https://img.shields.io/badge/python-3.9%2B-blue)
![OpenCV](https://img.shields.io/badge/OpenCV-contrib-green)

![demo](assets/demo.gif)

Facetion shows each person's name next to their face in your webcam video. It takes three steps:

1. **Capture:** the program takes 50 photos of your face.
2. **Train:** the program learns what your face looks like from those photos.
3. **Recognize:** the program shows your name on screen when it sees you.

Everything stays on your computer. No photos are uploaded anywhere.

---

## Table of Contents

- [What you need](#what-you-need)
- [Beginner's guide](#beginners-guide)
  - [Step 1: Open a terminal](#step-1-open-a-terminal)
  - [Step 2: Install Python](#step-2-install-python)
  - [Step 3: Download Facetion](#step-3-download-facetion)
  - [Step 4: Set up Facetion (one time only)](#step-4-set-up-facetion-one-time-only)
  - [Step 5: Use Facetion](#step-5-use-facetion)
  - [Next time you use it](#next-time-you-use-it)
- [Common problems](#common-problems)
- [For developers](#for-developers)

---

## What you need

- A computer running **Windows**, **macOS**, or **Linux**
- A **webcam** (a built-in laptop camera is fine)
- An internet connection for the one-time setup
- About **15 minutes**

You don't need any programming experience. You'll copy and paste a few commands, and this guide explains each one.

---

## Beginner's guide

### Step 1: Open a terminal

A *terminal* is a window where you type commands for your computer. You'll use it to install and run Facetion.

| Your computer | How to open the terminal |
|---|---|
| **Windows** | Press the <kbd>Windows</kbd> key, type **cmd**, and press <kbd>Enter</kbd>. A black window called *Command Prompt* opens. |
| **macOS** | Press <kbd>Cmd</kbd> + <kbd>Space</kbd>, type **Terminal**, and press <kbd>Enter</kbd>. |
| **Linux** | Press <kbd>Ctrl</kbd> + <kbd>Alt</kbd> + <kbd>T</kbd>. |

> 💡 **How to run a command:** copy the command from this guide, paste it into the terminal, and press <kbd>Enter</kbd>. Wait until it finishes before running the next one.
>
> To paste: use <kbd>Ctrl</kbd> + <kbd>V</kbd> on Windows (or right-click), <kbd>Cmd</kbd> + <kbd>V</kbd> on macOS, and <kbd>Ctrl</kbd> + <kbd>Shift</kbd> + <kbd>V</kbd> on Linux.

### Step 2: Install Python

Facetion is written in Python, so your computer needs Python to run it. You might already have it. To check, run:

| Windows | macOS / Linux |
|---|---|
| `python --version` | `python3 --version` |

- ✅ If you see something like **`Python 3.12.4`**, and the number after `3.` is **9 or higher**, you already have Python. Go to [Step 3](#step-3-download-facetion).
- ❌ If you see an error such as *"not recognized"* or *"command not found"*, or the version is lower, install Python:

**Windows**
1. Go to [python.org/downloads](https://www.python.org/downloads/) and click the big yellow **Download Python** button.
2. Open the file you downloaded.
3. ⚠️ **Important:** on the first screen, tick the box **"Add python.exe to PATH"** at the bottom.
4. Click **Install Now** and wait for it to finish.

**macOS**
1. Go to [python.org/downloads](https://www.python.org/downloads/) and click **Download Python**.
2. Open the downloaded `.pkg` file and click **Continue** until the installation finishes.

**Linux (Ubuntu / Debian)**
```bash
sudo apt update && sudo apt install python3 python3-pip python3-venv
```
It will ask for your computer password. Nothing appears on screen while you type it. That's normal.

When the installation finishes, **close the terminal, open a new one**, and run the version check again to make sure it worked.

### Step 3: Download Facetion

**The easy way (no extra software):**

1. Go to the project page: [github.com/dzungluu25/Facetion](https://github.com/dzungluu25/Facetion)
2. Click the green **`<> Code`** button, then click **Download ZIP**.
3. Find the downloaded file (usually in your **Downloads** folder) and unzip it:
   - **Windows:** right-click it and choose **Extract All… → Extract**
   - **macOS:** double-click it
4. You now have a folder called **`Facetion-master`**. Move it somewhere easy to find, such as your **Desktop**.

<details>
<summary>Alternative: download with Git (if you already use Git)</summary>

```bash
git clone https://github.com/dzungluu25/Facetion.git
```

If Git isn't installed, get it from [git-scm.com/downloads](https://git-scm.com/downloads). On macOS you can run `xcode-select --install` instead.
</details>

### Step 4: Set up Facetion (one time only)

You only need to do this step once.

**4a. Go into the Facetion folder in your terminal.**

Type `cd` followed by a space. **Don't press Enter yet.** Then drag the Facetion folder from your desktop into the terminal window, and the folder's location will be filled in. Now press <kbd>Enter</kbd>.

It should look something like this:

```bash
cd /Users/yourname/Desktop/Facetion-master        # macOS / Linux
cd C:\Users\yourname\Desktop\Facetion-master      # Windows
```

**4b. Create a private workspace for Facetion** (this keeps it separate from the rest of your computer):

| Windows | macOS / Linux |
|---|---|
| `python -m venv venv` | `python3 -m venv venv` |

**4c. Turn on the workspace:**

| Windows | macOS / Linux |
|---|---|
| `venv\Scripts\activate` | `source venv/bin/activate` |

✅ The start of the line in your terminal should now show **`(venv)`**.

**4d. Install the parts Facetion needs:**

```bash
pip install -r requirements.txt
```

This downloads a few files and can take a few minutes. When the terminal is ready for a new command again, setup is done. 🎉

### Step 5: Use Facetion

Make sure your terminal still shows **`(venv)`**. If it doesn't, see [Next time you use it](#next-time-you-use-it).

#### ① Add a person

```bash
python -m facetion capture
```

1. Type the person's name and press <kbd>Enter</kbd>.
2. A camera window opens. **Look at the camera** and slowly turn your head a little left, right, up, and down.
3. After 50 photos the window closes by itself. Press <kbd>Esc</kbd> to stop early.

> 📷 The first time, your computer may ask for permission to use the camera. Click **Allow**.

To add more people, run this command again for each person.

#### ② Teach Facetion the faces

```bash
python -m facetion train
```

This takes a few seconds. You'll see a message like `2 users trained`.

> Run this again every time you add a new person.

#### ③ Start recognizing faces

```bash
python -m facetion recognize
```

A camera window opens. Each face gets a green box with a name:

- **Your name and a percentage:** Facetion recognized you. A higher percentage means it's more sure.
- **"Who are you?":** Facetion doesn't know this face. Add the person with step ①, then repeat step ②.

Press <kbd>Esc</kbd> to close the window.

### Next time you use it

You don't need to repeat the setup. Open a terminal and run:

**Windows**
```bat
cd C:\Users\yourname\Desktop\Facetion-master
venv\Scripts\activate
python -m facetion recognize
```

**macOS / Linux**
```bash
cd ~/Desktop/Facetion-master
source venv/bin/activate
python -m facetion recognize
```

Change the folder path in the first line to where you put Facetion.

---

## Common problems

| What you see | What to do |
|---|---|
| `'python' is not recognized…` or `command not found` | Python isn't installed, or Windows can't find it. Reinstall Python and tick **"Add python.exe to PATH"** (see [Step 2](#step-2-install-python)), then close and reopen the terminal. |
| `No such file or directory: 'requirements.txt'` | You're not inside the Facetion folder. Repeat [Step 4a](#step-4-set-up-facetion-one-time-only). |
| `No module named facetion` or `No module named cv2` | The workspace isn't turned on, or you're in the wrong folder. Go to the Facetion folder and run the activate command from [Step 4c](#step-4-set-up-facetion-one-time-only). |
| PowerShell says *"running scripts is disabled"* | Use **Command Prompt** (`cmd`) instead, or run `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned` once. |
| The camera window is black or doesn't open | Close other apps using the camera (Zoom, Teams, etc.). On macOS, go to **System Settings → Privacy & Security → Camera** and allow **Terminal**. |
| `Model not found` | Run step ② `train` before step ③ `recognize`. |
| `No samples found` | Run step ① `capture` before step ② `train`. |
| It shows the wrong name or "Who are you?" for a known person | Use good lighting and face the camera. Run `capture` again for that person to add more photos, then run `train` again. |
| `module 'cv2' has no attribute 'face'` | Run `pip uninstall -y opencv-python opencv-contrib-python`, then `pip install -r requirements.txt`. |

---

## For developers

### How it works

Faces are detected with a Haar cascade and identified with OpenCV's **LBPH** (Local Binary Patterns Histograms) recognizer. Everything runs on the CPU.

```
 capture                train                    recognize
┌──────────────┐      ┌───────────────┐        ┌──────────────────┐
│ Webcam →     │      │ data/images/→ │        │ Webcam →         │
│ detect face →│ ───▶ │ LBPH training │ ─────▶ │ detect + predict │
│ data/images/ │      │ →trainer.yml  │        │ → name on screen │
└──────────────┘      └───────────────┘        └──────────────────┘
        └──── data/names.json (ID → name) ──────────────┘
```

### Quick start

```bash
git clone https://github.com/dzungluu25/Facetion.git && cd Facetion
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt
python -m facetion capture | train | recognize
```

### Project structure

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
│   ├── images/          # (generated, git-ignored) face samples: user-<id>-<n>.jpg
│   └── trainer.yml      # (generated, git-ignored) trained model
├── assets/demo.gif
├── requirements.txt
└── README.md
```

### Configuration

All settings live in [`facetion/config.py`](facetion/config.py):

| Setting | Default | Description |
|---|---|---|
| `SAMPLES_PER_USER` | `50` | Face images captured per person |
| `CAMERA_INDEX` | `0` | Change this if you have several cameras |
| `FRAME_WIDTH` / `FRAME_HEIGHT` | `640×480` | Camera resolution |
| `CONFIDENCE_THRESHOLD` | `51` | Max LBPH distance accepted as a match (lower = stricter) |

### Known limitations

- Haar cascades work best on frontal, well-lit faces.
- LBPH is a classical algorithm. For higher accuracy, consider deep-learning embeddings (e.g. `face_recognition`, InsightFace).

### Contributing

1. Fork the repo
2. Create a branch: `git checkout -b feature/my-feature`
3. Commit your changes and open a Pull Request

## Acknowledgements

- [OpenCV](https://opencv.org/) for the Haar cascade and LBPH face recognizer
