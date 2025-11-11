import cv2
import json
import time
import numpy as np
from pathlib import Path

KNOWN_DIR = Path("known_faces")
MODEL_DIR = Path("models")
MODEL_DIR.mkdir(parents=True, exist_ok=True)
KNOWN_DIR.mkdir(parents=True, exist_ok=True)  # ensure it exists at startup

CASCADE_PATH = Path(cv2.data.haarcascades) / "haarcascade_frontalface_alt2.xml"
MODEL_PATH = MODEL_DIR / "lbph_face.yml"
LABELS_PATH = MODEL_DIR / "lbph_labels.json"
# NEW: profile cascade for side faces
PROFILE_CASCADE_PATH = Path(cv2.data.haarcascades) / "haarcascade_profileface.xml"

# LBPH distance threshold: lower is better. Tune 45–80 depending on your data.
LBPH_THRESHOLD = 70
FACE_SIZE = (128, 128)
ROLL_FALLBACK_ANGLES = [-20, -10, 10, 20]

# Augmentation settings
AUGMENT_FLIP = True
AUGMENT_GAMMAS = [0.6, 0.8, 1.2, 1.4]  # <1 brightens (dark images), >1 darkens (bright images)

# Initialize detectors once (faster than reloading every frame)
FRONTAL_DETECTOR = cv2.CascadeClassifier(str(CASCADE_PATH))
if FRONTAL_DETECTOR.empty():
    raise SystemExit(f"Failed to load frontal cascade at {CASCADE_PATH}")

PROFILE_DETECTOR = cv2.CascadeClassifier(str(PROFILE_CASCADE_PATH))
if PROFILE_DETECTOR.empty():
    PROFILE_DETECTOR = None  # fallback if profile cascade is missing

def ensure_cv2_face():
    if not hasattr(cv2, "face"):
        raise SystemExit("OpenCV 'face' module missing. Install: pip install opencv-contrib-python")

# Simple IoU/NMS to dedupe overlapping detections
def _iou(a, b):
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    x1, y1 = max(ax, bx), max(ay, by)
    x2, y2 = min(ax + aw, bx + bw), min(ay + ah, by + bh)
    iw, ih = max(0, x2 - x1), max(0, y2 - y1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    union = aw * ah + bw * bh - inter
    return inter / union

def _nms(boxes, iou_thresh=0.35):
    if not boxes:
        return []
    boxes = sorted(boxes, key=lambda b: b[2] * b[3], reverse=True)
    keep = []
    for b in boxes:
        if all(_iou(b, k) < iou_thresh for k in keep):
            keep.append(b)
    return keep

# --- Photometric normalization and augmentation ------------------------------

def _tan_triggs(gray, gamma=0.2, sigma0=1.0, sigma1=2.0, tau=10.0, alpha=0.1):
    # Inputs/outputs are uint8; internal is float32
    I = gray.astype(np.float32) / 255.0
    # Gamma correction
    I = np.power(np.maximum(I, 1e-6), gamma)
    # Difference of Gaussians (DoG)
    G0 = cv2.GaussianBlur(I, (0, 0), sigma0)
    G1 = cv2.GaussianBlur(I, (0, 0), sigma1)
    I = G0 - G1
    # Contrast equalization (two normalization steps)
    eps = 1e-6
    value = np.power(np.abs(I), alpha).mean() ** (1.0 / alpha)
    I = I / (value + eps)
    value = np.power(np.minimum(tau, np.abs(I)), alpha).mean() ** (1.0 / alpha)
    I = I / (value + eps)
    # Truncate and rescale
    I = tau * np.tanh(I / tau)
    I = cv2.normalize(I, None, 0, 255, cv2.NORM_MINMAX)
    return I.astype(np.uint8)

def _gamma_lut(gamma: float):
    x = np.arange(256, dtype=np.float32) / 255.0
    y = np.clip(np.power(x, gamma) * 255.0, 0, 255)
    return y.astype(np.uint8)

def augment_face(face_u8):
    aug = [face_u8]
    # gamma variants (lighting)
    for g in AUGMENT_GAMMAS:
        aug.append(cv2.LUT(face_u8, _gamma_lut(g)))
    # optional horizontal flips
    if AUGMENT_FLIP:
        aug.extend([cv2.flip(a, 1) for a in list(aug)])
    return aug

# ---------------------------------------------------------------------------

def _expand_box(x, y, w, h, scale, W, H):
    cx, cy = x + w / 2.0, y + h / 2.0
    nw, nh = w * scale, h * scale
    nx = int(max(0, round(cx - nw / 2.0)))
    ny = int(max(0, round(cy - nh / 2.0)))
    nx2 = int(min(W, round(cx + nw / 2.0)))
    ny2 = int(min(H, round(cy + nh / 2.0)))
    return nx, ny, max(0, nx2 - nx), max(0, ny2 - ny)

def _rect_corners(x, y, w, h):
    return np.array([
        [x,     y],
        [x + w, y],
        [x,     y + h],
        [x + w, y + h]
    ], dtype=np.float32)

def _apply_affine(M2x3, ptsNx2):
    pts_h = np.hstack([ptsNx2, np.ones((ptsNx2.shape[0], 1), dtype=np.float32)])
    out = pts_h @ M2x3.T
    return out

def _detect_frontal_on_rotations(gray, angles):
    H, W = gray.shape[:2]
    boxes = []
    for ang in angles:
        M = cv2.getRotationMatrix2D((W / 2.0, H / 2.0), ang, 1.0)
        rot = cv2.warpAffine(gray, M, (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        f = FRONTAL_DETECTOR.detectMultiScale(rot, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60))
        if f is None or len(f) == 0:
            continue
        M_inv = cv2.invertAffineTransform(M)
        try:
            rects = f.tolist()
        except Exception:
            rects = list(f)
        for (x, y, w, h) in rects:
            pts = _rect_corners(x, y, w, h)
            pts2 = _apply_affine(M_inv, pts)
            x1, y1 = np.floor(pts2.min(axis=0)).astype(int)
            x2, y2 = np.ceil(pts2.max(axis=0)).astype(int)
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(W, x2), min(H, y2)
            bw, bh = max(0, x2 - x1), max(0, y2 - y1)
            if bw >= 40 and bh >= 40:
                boxes.append((int(x1), int(y1), int(bw), int(bh)))
    return boxes

def detect_faces(gray):
    # Frontal
    boxes = []
    f = FRONTAL_DETECTOR.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60))
    if f is not None and len(f):
        try:
            boxes.extend(f.tolist())
        except Exception:
            boxes.extend(list(f))

    # Profile (right-facing in image space)
    if PROFILE_DETECTOR is not None:
        p = PROFILE_DETECTOR.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60))
        if p is not None and len(p):
            try:
                boxes.extend(p.tolist())
            except Exception:
                boxes.extend(list(p))

        # Detect left-facing by mirroring
        flipped = cv2.flip(gray, 1)
        pf = PROFILE_DETECTOR.detectMultiScale(flipped, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60))
        if pf is not None and len(pf):
            w_img = gray.shape[1]
            try:
                pf_iter = pf.tolist()
            except Exception:
                pf_iter = list(pf)
            for (x, y, w, h) in pf_iter:
                boxes.append((w_img - x - w, y, w, h))

    # NEW: if nothing found, try small roll angles
    if not boxes:
        boxes.extend(_detect_frontal_on_rotations(gray, ROLL_FALLBACK_ANGLES))

    # Normalize and dedupe
    boxes = [tuple(map(int, b)) for b in boxes]
    boxes = _nms(boxes, iou_thresh=0.35)
    return boxes

def preprocess_face(gray_roi):
    # Resize, Tan–Triggs (illumination invariance), then light CLAHE
    face = cv2.resize(gray_roi, FACE_SIZE, interpolation=cv2.INTER_CUBIC)
    face = _tan_triggs(face, gamma=0.2, sigma0=1.0, sigma1=2.0, tau=10.0, alpha=0.1)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    face = clahe.apply(face)
    return face

# NEW: cleanup utility to auto-remove images without faces
def clean_known_faces(delete_empty_dirs=True):
    if not KNOWN_DIR.exists():
        return 0, 0
    removed = 0
    emptied = 0
    for person_dir in sorted(p for p in KNOWN_DIR.iterdir() if p.is_dir()):
        for img_path in sorted(person_dir.glob("*.*")):
            img = cv2.imread(str(img_path))
            if img is None:
                if img_path.exists():
                    img_path.unlink()
                    removed += 1
                continue
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            faces = detect_faces(gray)
            if len(faces) == 0:
                if img_path.exists():
                    img_path.unlink()
                    removed += 1
        # remove empty person directory
        try:
            next(person_dir.iterdir())
        except StopIteration:
            if delete_empty_dirs:
                person_dir.rmdir()
                emptied += 1
    print(f"Cleaned known_faces: removed {removed} images, removed {emptied} empty folders.")
    return removed, emptied

def train_lbph_from_known():
    ensure_cv2_face()
    clean_known_faces(delete_empty_dirs=True)

    images, labels = [], []
    name_to_id = {}
    next_id = 0

    if not KNOWN_DIR.exists():
        print(f"Create {KNOWN_DIR}/<person>/image.jpg first.")
        return None, None

    for person_dir in sorted(p for p in KNOWN_DIR.iterdir() if p.is_dir()):
        name = person_dir.name
        if name not in name_to_id:
            name_to_id[name] = next_id
            next_id += 1

        for img_path in sorted(person_dir.glob("*.*")):
            img = cv2.imread(str(img_path))
            if img is None:
                print(f"[skip] cannot read {img_path}")
                continue
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            faces = detect_faces(gray)

            # If images are already cropped faces and detector fails, use whole image
            if len(faces) == 0:
                h, w = gray.shape[:2]
                if abs(w - FACE_SIZE[0]) <= 10 and abs(h - FACE_SIZE[1]) <= 10:
                    faces = [(0, 0, w, h)]
                else:
                    print(f"[remove] no face in {img_path}")
                    try:
                        img_path.unlink()
                    except Exception:
                        pass
                    continue

            # largest face
            x, y, w, h = max(faces, key=lambda b: b[2] * b[3])
            H, W = gray.shape[:2]
            x, y, w, h = _expand_box(x, y, w, h, 1.20, W, H)
            face = preprocess_face(gray[y:y+h, x:x+w])

            for a in augment_face(face):
                images.append(a)
                labels.append(name_to_id[name])

    if not images:
        print("No training data found.")
        return None, None

    recognizer = cv2.face.LBPHFaceRecognizer_create(radius=1, neighbors=8, grid_x=8, grid_y=8)
    recognizer.train(images, np.array(labels))
    recognizer.save(str(MODEL_PATH))
    LABELS_PATH.write_text(json.dumps(name_to_id, indent=2))
    print(f"Trained LBPH on {len(images)} faces for {len(name_to_id)} people.")
    return recognizer, {v: k for k, v in name_to_id.items()}

def load_or_train():
    ensure_cv2_face()
    if MODEL_PATH.exists() and LABELS_PATH.exists():
        recognizer = cv2.face.LBPHFaceRecognizer_create()
        recognizer.read(str(MODEL_PATH))
        try:
            labels = json.loads(LABELS_PATH.read_text())
            # support both schemas:
            # 1) {"0": "Alice"} or 2) {"Alice": 0}
            if labels and all(isinstance(k, str) and k.isdigit() for k in labels.keys()):
                id_to_name = {int(k): v for k, v in labels.items()}
            else:
                id_to_name = {int(v): k for k, v in labels.items()}
        except Exception:
            print("Labels file invalid; retraining.")
            return train_lbph_from_known()
        print(f"Loaded model with {len(id_to_name)} people.")
        return recognizer, id_to_name
    return train_lbph_from_known()

def enroll_from_camera(name: str, cap, samples: int = 25, interval: float = 1.0):
    """Capture 'samples' normalized face crops into known_faces/<name>/"""
    person_dir = KNOWN_DIR / name
    person_dir.mkdir(parents=True, exist_ok=True)
    saved = 0
    last_save = 0.0
    print(f"Enrolling '{name}'. Look at the camera. SPACE=manual save, ESC=cancel.")
    while saved < samples:
        ok, frame = cap.read()
        if not ok:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = detect_faces(gray)
        if len(faces):
            x, y, w, h = max(faces, key=lambda b: b[2]*b[3])
            H, W = gray.shape[:2]
            x2, y2, w2, h2 = _expand_box(x, y, w, h, 1.15, W, H)
            crop = preprocess_face(gray[y2:y2+h2, x2:x2+w2])

            # draw box for feedback
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 255), 2)

            now = time.time()
            if now - last_save >= interval:
                out = person_dir / f"{int(time.time()*1000)}_{saved:02d}.png"
                cv2.imwrite(str(out), crop)  # save normalized face (grayscale)
                saved += 1
                last_save = now

            cv2.putText(frame, f"Saved {saved}/{samples}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,255,255), 2)
        else:
            cv2.putText(frame, "No face detected", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)

        cv2.imshow("Enroll", frame)
        k = cv2.waitKey(10) & 0xFF
        if k == 27:  # ESC
            break
        if k == 32:  # SPACE manual save
            if len(faces):
                out = person_dir / f"{int(time.time()*1000)}_{saved:02d}.png"
                cv2.imwrite(str(out), crop)
                saved += 1
                last_save = time.time()
    cv2.destroyWindow("Enroll")
    print(f"Saved {saved} images to {person_dir}")

def main():
    rec, id_to_name = load_or_train()
    if rec is None:
        print("No model yet. Press 'e' to enroll someone.")

    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if not cap.isOpened():
        raise RuntimeError("Cannot open camera. Try index 1/2 or close other apps.")

    print("Keys: e=enroll, r=retrain, c=clean, a=add current sample to recognized, q=quit")

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = detect_faces(gray)

        best = None  # (x,y,w,h,name,dist)
        if 'rec' in locals() and rec is not None:
            for (x, y, w, h) in faces:
                H, W = gray.shape[:2]
                x2, y2, w2, h2 = _expand_box(x, y, w, h, 1.15, W, H)
                roi = preprocess_face(gray[y2:y2+h2, x2:x2+w2])
                label, dist = rec.predict(roi)
                name = id_to_name.get(label, "Unknown")
                if dist > LBPH_THRESHOLD:
                    name = "Unknown"
                if best is None or w*h > best[2]*best[3]:
                    best = (x, y, w, h, name, dist, (x2, y2, w2, h2), roi)
                color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)
                cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
                cv2.putText(frame, f"{name} ({dist:.1f})", (x, max(0, y-8)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        cv2.imshow("LBPH face recognition (local-only)", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        if key == ord('e'):
            cv2.destroyWindow("LBPH face recognition (local-only)")
            person = input("Name to enroll: ").strip()
            if person:
                enroll_from_camera(person, cap, samples=25, interval=1.0)
            cv2.namedWindow("LBPH face recognition (local-only)")
        if key == ord('r'):
            print("Cleaning known_faces, then retraining...")
            clean_known_faces(delete_empty_dirs=True)
            rec2, id_to_name2 = train_lbph_from_known()
            if rec2:
                rec, id_to_name = rec2, id_to_name2
                print("Retrained.")
        if key == ord('c'):
            clean_known_faces(delete_empty_dirs=True)
        if key == ord('a') and best is not None:
            x, y, w, h, name, dist, (x2, y2, w2, h2), roi = best
            if name != "Unknown" and dist <= LBPH_THRESHOLD:
                person_dir = KNOWN_DIR / name
                person_dir.mkdir(parents=True, exist_ok=True)
                out = person_dir / f"online_{int(time.time()*1000)}.png"
                cv2.imwrite(str(out), roi)
                print(f"Added sample for '{name}': {out}. Press 'r' to retrain.")
            else:
                print("No confident recognition to add. Move closer or improve lighting.")

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()