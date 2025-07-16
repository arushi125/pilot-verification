#!pip install deepface --proxy=172.16.1.61:8080
#!pip install tf-keras --proxy=172.16.1.61:8080
#!pip install tensorflow --proxy=172.16.1.61:8080
#!pip install --upgrade deepface
from deepface import DeepFace

model = DeepFace.build_model("Facenet")

result = DeepFace.verify(
    img1_path="CSMT4182_F.jpg",
    img2_path="CSMT4182_S1.jpg",
    model_name="Facenet",
    model=model   # use preloaded model to skip reload
)

print(result)

!pip uninstall deepface -y --proxy=172.16.1.61:8080
!pip install --no-cache-dir deepface --proxy=172.16.1.61:8080
import deepface
print(deepface.__version__)
import cv2
import numpy as np
from deepface import DeepFace
from numpy.linalg import norm
from matplotlib import pyplot as plt

# Step 1: Load DeepFace model wrapper (Facenet)
print("LOADING Model")
model_wrapper = DeepFace.build_model("Facenet")
model = model_wrapper.model  # Get the underlying Keras model
print("Model LOADED")

# Step 2: Preprocess face images manually
def preprocess_face(image_path, target_size=(160, 160)):
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Could not read image: {image_path}")
    
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    face = cv2.resize(img, target_size)
    face = face.astype(np.float32) / 255.0
    face = np.expand_dims(face, axis=0)
    return face
# Step 3: Preprocess both images
#img1 = preprocess_face("CSMT4182_F.jpg")
img1 = preprocess_face("PUNE1409_F.jpg")
img1 = preprocess_face("CSMT4182_S1.jpg")
#img1 = preprocess_face("CSMT4182_S6.jpg")
img2 = preprocess_face("PUNE1409_S1.jpg")
#img1 = preprocess_face("PUNE1409_S2.jpg")

fig, axs = plt.subplots(1, 2, figsize=(10, 5))
img1 = cv2.cvtColor(cv2.imread("PUNE1409_F.jpg"), cv2.COLOR_BGR2RGB)
img2 = cv2.cvtColor(cv2.imread("PUNE1409_S1.jpg"), cv2.COLOR_BGR2RGB)

axs[0].imshow(img1)
axs[0].set_title("Front")
axs[0].axis("off")

axs[1].imshow(img2)
axs[1].set_title("Side")
axs[1].axis("off")

plt.show()
# Step 4: Generate embeddings
embedding1 = model.predict(img1)[0]
embedding2 = model.predict(img2)[0]

# Step 5: Calculate L2 distance
distance = norm(embedding1 - embedding2)
threshold = 5  # typical Facenet threshold

print(f"Distance = {distance:.4f}")
print("Same person" if distance < threshold else "Different people")
import cv2
import numpy as np
from deepface import DeepFace
from numpy.linalg import norm
import matplotlib.pyplot as plt

# Load Facenet model
model_wrapper = DeepFace.build_model("Facenet")
model = model_wrapper.model

def preprocess_face(image_path, target_size=(160, 160)):
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Could not read image: {image_path}")
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img_rgb, target_size)
    img_normalized = img_resized.astype(np.float32) / 255.0
    img_expanded = np.expand_dims(img_normalized, axis=0)
    return img_expanded, img_rgb  # return both for visualization

# Preprocess images
img1, img1_show = preprocess_face("CSMT4182_S1.jpg")
img2, img2_show = preprocess_face("PUNE1409_S1.jpg")
embedding1 = model.predict(img1)[0]
embedding2 = model.predict(img2)[0]

# Calculate distance
distance = norm(embedding1 - embedding2)
threshold = 3  # typical threshold for Facenet

# Show result
print(f"Distance: {distance:.4f}")
print("✅ Same person" if distance < threshold else "❌ Different people")

# Visualize both images
fig, axes = plt.subplots(1, 2, figsize=(8, 4))
axes[0].imshow(img1_show)
axes[0].set_title("Image 1")
axes[0].axis("off")
axes[1].imshow(img2_show)
axes[1].set_title("Image 2")
axes[1].axis("off")
plt.suptitle(f"Distance: {distance:.4f} → {'Same' if distance < threshold else 'Different'}")
plt.show()
mport cv2
import numpy as np
from deepface import DeepFace
from numpy.linalg import norm
import matplotlib.pyplot as plt

# Load ArcFace model (you can switch back to Facenet if needed)
model_wrapper = DeepFace.build_model("ArcFace")
model = model_wrapper.model

# Face detector and embedding input config
def detect_align(img_path):
    faces = DeepFace.extract_faces(
        img_path=img_path,
        detector_backend="opencv",  # Try "retinaface" if accuracy matters
        enforce_detection=False     #  allow weak or partial detection
    )
    if len(faces) == 0:
        raise ValueError(f"No face detected in {img_path}")
    face_rgb = faces[0]["face"]
    face_input = face_rgb.astype("float32") / 127.5 - 1
    return np.expand_dims(face_input, axis=0), face_rgb

# Load and align images
img1_tensor, img1_show = detect_align("CSMT4182_S1.jpg")
img2_tensor, img2_show = detect_align("PUNE1409_S1.jpg")

# Predict embeddings
embedding1 = model.predict(img1_tensor)[0]
embedding2 = model.predict(img2_tensor)[0]

# Cosine similarity
cos_sim = np.dot(embedding1, embedding2) / (norm(embedding1) * norm(embedding2))
threshold = 0.35  # for ArcFace, adjust if needed

# Result
print(f"Cosine Similarity: {cos_sim:.4f}")
print(" Same person" if cos_sim > threshold else " Different people")

# Show images
fig, ax = plt.subplots(1, 2, figsize=(8, 4))
ax[0].imshow(img1_show)
ax[0].set_title("Image 1")
ax[0].axis("off")
ax[1].imshow(img2_show)
ax[1].set_title("Image 2")
ax[1].axis("off")
plt.suptitle(f"Cosine Similarity: {cos_sim:.4f} → {'Same' if cos_sim > threshold else 'Different'}")
plt.show()

