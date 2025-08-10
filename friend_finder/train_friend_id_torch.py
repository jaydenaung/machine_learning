# Friend Finder App by Jayden Aung
# train_friend_id_torch.py 
# Learn a "friend" prototype embedding + a decision threshold (cosine similarity).

import os, glob
import numpy as np
from PIL import Image
import torch
from facenet_pytorch import MTCNN, InceptionResnetV1

FRIEND_DIR = "data/friend"
NOT_DIR    = "data/not_friend"
OUT_PATH   = "friend_proto.npz"

# Embedder on MPS if available; keep MTCNN on CPU (workaround for MPS pooling bug)
device = "mps" if torch.backends.mps.is_available() else "cpu"

# Detector/aligner on CPU
mtcnn = MTCNN(image_size=160, margin=20, post_process=True, device="cpu")

# Embedding model on MPS/CPU
resnet = InceptionResnetV1(pretrained='vggface2').eval().to(device)

def embed_image(path: str):
    try:
        img = Image.open(path).convert("RGB")
    except Exception:
        return None
    with torch.no_grad():
        face, prob = mtcnn(img, return_prob=True)
        if face is None:
            return None
        emb = resnet(face.unsqueeze(0).to(device)).cpu().numpy()[0]
        emb = emb / (np.linalg.norm(emb) + 1e-9)
        return emb

def cos(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / ((np.linalg.norm(a)*np.linalg.norm(b)) + 1e-9))

def main():
    exts = (".jpg", ".jpeg", ".png", ".webp")
    friend_imgs = [p for p in glob.glob(os.path.join(FRIEND_DIR, "*")) if p.lower().endswith(exts)]
    not_imgs    = [p for p in glob.glob(os.path.join(NOT_DIR,    "*")) if p.lower().endswith(exts)]
    assert len(friend_imgs) >= 8,  "Need at least ~8 images in data/friend"
    assert len(not_imgs)    >= 20, "Need at least ~20 images in data/not_friend"
    print(f"Found friend={len(friend_imgs)}, not_friend={len(not_imgs)}")

    friend_vecs = [embed_image(p) for p in friend_imgs]
    friend_vecs = np.array([v for v in friend_vecs if v is not None])
    not_vecs    = [embed_image(p) for p in not_imgs]
    not_vecs    = np.array([v for v in not_vecs if v is not None])

    print(f"Embedded friend={len(friend_vecs)}, not_friend={len(not_vecs)}")
    assert len(friend_vecs) >= 6 and len(not_vecs) >= 15, "Too many failed detections—check images."

    # Friend prototype (mean embedding, normalized)
    proto = friend_vecs.mean(axis=0)
    proto = proto / (np.linalg.norm(proto) + 1e-9)

    # Similarities
    pos = np.array([cos(proto, v) for v in friend_vecs])
    neg = np.array([cos(proto, v) for v in not_vecs])

    # Threshold search (favor precision to avoid false “friend”)
    thr_grid = np.linspace(0.55, 0.95, 81)
    best_score, best_thr = -1e9, 0.8
    for thr in thr_grid:
        tp = (pos >= thr).sum(); fn = (pos < thr).sum()
        fp = (neg >= thr).sum(); tn = (neg < thr).sum()
        prec = tp / (tp + fp + 1e-9)
        rec  = tp / (tp + fn + 1e-9)
        beta = 0.5  # precision-weighted
        fbeta = (1+beta**2) * (prec*rec) / (beta**2*prec + rec + 1e-9)
        score = fbeta - 0.01*fp  # tiny penalty for any FP
        if score > best_score:
            best_score, best_thr = score, thr

    print(f"Chosen threshold: {best_thr:.3f}")
    print(f"Pos mean={pos.mean():.3f}  Neg mean={neg.mean():.3f}")

    np.savez_compressed(
        OUT_PATH,
        proto=proto.astype(np.float32),
        thr=np.array([best_thr], dtype=np.float32),
        stats=np.array([pos.mean(), neg.mean()], dtype=np.float32),
    )
    print(f"Saved {OUT_PATH}. Next: python friend_id_app_torch.py")

if __name__ == "__main__":
    main()
