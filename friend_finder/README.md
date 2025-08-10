# Friend Finder by Jayden Aung

Friend Finder is a simple machine learning demo that uses **facenet-pytorch** to:
- Train a facial embedding of a specific **friend** (from your provided photos)
- Compare any new photo against this learned representation
- Predict whether the person is your friend or not, with a confidence score

## 📂 Project Structure
```bash
friend_finder/
│
├── train_friend_id_torch.py # Train the friend embedding & threshold
├── friend_id_app_torch.py # Gradio web app for recognition
├── requirements.txt # Python dependencies
│
└── data/
├── friend/ # images of your friend (≥ 8 images)
└── not_friend/ # images of other people (≥ 20 images)
```

## ⚙️ Setup

1. **Clone the repository**
```bash
git clone https://github.com/<your-username>/friend_finder.git
cd friend_finder

2. **Create a virtual environment
```bash
python3 -m venv venv
source venv/bin/activate  # macOS / Linux
```

# Or
```bash
venv\Scripts\activate     # Windows
```

3. Install dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

4. Prepare your data

```bash
data/
├── friend/        # Put at least 8 clear face photos of your friend
└── not_friend/    # Put at least 20 face photos of other people
```
* Photos should be in .jpg, .jpeg, .png, or .webp format.

* Face should be visible and not too small.

##  Training
Run the training script to:

* Compute your friend's prototype face embedding
* Find an optimal decision threshold
* Save them into friend_proto.npz

If training is successful, you’ll see output like:

```
Found friend=10, not_friend=25
Embedded friend=10, not_friend=25
Chosen threshold: 0.780
Saved friend_proto.npz
```

## 🚀 Running the Web App
Launch the Gradio interface:

```bash
python friend_id_app_torch.py
```

You can:

* Upload a photo
* Use your webcam (if supported)
* See prediction + confidence score

Example output:

```
# In friend_id_app_torch.py, change:
demo.launch()
# To:
demo.launch(share=True)
```

## Requirements
Python 3.8+

* torch
* torchvision
* facenet-pytorch
* gradio
* pillow
* numpy


Install all at once:

```bash
pip install -r requirements.txt
```

## 🔒 Security Note
This is a demo project, not production-grade biometric authentication.

Never use for real-world security or identity verification without proper consent, privacy safeguards, and model evaluation.

Store personal images securely.

The app is only as good as its training data — poor quality or insufficient images will cause misclassification.


