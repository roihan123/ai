# AI (LBPH Face Recognition)

Local-only face detection and recognition using OpenCV LBPH.
- Frontal + profile (side) detection with mirrored pass and small roll fallback.
- Robust preprocessing (Tan–Triggs + CLAHE).
- Training augmentation (flips + gamma) for dark/bright images.
- Auto-clean images without faces.
- Live dataset growth from camera.

## Requirements
- Windows with Python 3.9–3.12
- Webcam
- pip can install opencv-contrib-python and numpy

## Quick start (Windows PowerShell)
1) Initialize environment and install deps:
```
.\setup.ps1
```
If scripts are blocked:
```
Set-ExecutionPolicy -Scope Process RemoteSigned
.\setup.ps1
```

2) Activate and run:
```
.\.venv\Scripts\Activate.ps1
python main.py
```

## Manual setup (alternative)
```
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python main.py
```

The app will auto-create folders:
- models/ for the trained model files
- known_faces/ for your training images

## Usage (keys in the main window)
- e: Enroll a new person (captures normalized face crops)
- r: Retrain model from images in known_faces
- c: Clean known_faces (remove images without detectable faces)
- a: Add current recognized face as an extra sample (then press r)
- q: Quit

Tips:
- During enrollment, look at the camera with slight left/right/tilt angles.
- If tilted faces read as “Unknown,” add a few tilted samples and retrain.
- LBPH threshold is configured in main.py (LBPH_THRESHOLD). Larger values accept more variation.

## Troubleshooting
- cv2.face not found: install contrib build
  ```
  pip install opencv-contrib-python --upgrade
  ```
- Camera busy or wrong index: edit VideoCapture(0) to 1 or 2.
- Large model files are not committed; models/ is ignored by Git. Rebuild locally with r.

## Project layout
```
ai/
  main.py
  requirements.txt
  setup.ps1
  known_faces/     # created on first run
  models/          # created on first run
```

## note
- this progeram is still doing locally and need a lot of dataset (faces) to make it work.
- program will train everytime it run and sometimes the program gonna be slow a little bit, depends on how many dataset in `known_faces`
