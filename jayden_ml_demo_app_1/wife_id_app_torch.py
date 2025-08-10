# wife_id_app_torch.py
# Gradio app using facenet-pytorch embeddings + learned threshold

import gradio as gr
import numpy as np
from PIL import Image
import torch
from facenet_pytorch import MTCNN, InceptionResnetV1

TITLE = "Jayden ML Demo – Wife Identifier (ML - PyTorch)"
DESC  = "Upload or use webcam. Uses face embeddings + a learned threshold. Tuned to avoid false 'wife'."

# Embedder on MPS if available; detector on CPU (workaround for MPS pooling bug)
device = "mps" if torch.backends.mps.is_available() else "cpu"
mtcnn = MTCNN(image_size=160, margin=20, post_process=True, device="cpu")
resnet = InceptionResnetV1(pretrained='vggface2').eval().to(device)

NPZ = np.load("wife_proto.npz")
PROTO = NPZ["proto"] / (np.linalg.norm(NPZ["proto"]) + 1e-9)
THR   = float(NPZ["thr"][0])

@torch.inference_mode()
def embed_rgb(np_img: np.ndarray):
    if np_img.dtype != np.uint8:
        np_img = np_img.astype(np.uint8)
    img = Image.fromarray(np_img)
    face, prob = mtcnn(img, return_prob=True)
    if face is None:
        raise RuntimeError("No face detected")
    emb = resnet(face.unsqueeze(0).to(device)).cpu().numpy()[0]
    emb = emb / (np.linalg.norm(emb) + 1e-9)
    return emb

def cosine(a, b):
    return float(np.dot(a, b) / ((np.linalg.norm(a)*np.linalg.norm(b)) + 1e-9))

def predict(np_img):
    if np_img is None:
        return "No image", {"wife": 0.0, "not_wife": 1.0}
    try:
        emb = embed_rgb(np_img)
    except Exception as e:
        return f"Face not found / error: {e}", {"wife": 0.0, "not_wife": 1.0}
    sim = cosine(PROTO, emb)
    is_wife = sim >= THR
    # Soft confidence around the threshold
    margin = 0.04
    conf_wife = float(1 / (1 + np.exp(-(sim - THR) / (margin + 1e-9))))
    conf = {"wife": conf_wife, "not_wife": 1.0 - conf_wife}
    label = f"{'✅ Wife' if is_wife else '❌ Not Wife'} (cos={sim:.3f}, thr={THR:.3f})"
    return label, conf

demo = gr.Interface(
    fn=predict,
    inputs=gr.Image(sources=["upload","webcam"], streaming=False, label="Upload or use webcam"),
    outputs=[gr.Label(label="Prediction"), gr.Label(num_top_classes=2, label="Confidence")],
    title=TITLE, description=DESC,
)

if __name__ == "__main__":
    # LAN access: demo.launch(server_name="0.0.0.0", server_port=7860)
    # Quick public HTTPS: demo.launch(share=True)
    demo.launch()

