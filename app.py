import streamlit as st # type: ignore
import torch # type: ignore
from torchvision.models import vit_b_16 # type: ignore
from torchvision import transforms # type: ignore
from PIL import Image # type: ignore
from collections import OrderedDict

# Fungsi untuk memuat model dari checkpoint
@st.cache_resource
def load_model():
    try:
        # Buat model ViT dengan arsitektur yang sama
        model = vit_b_16(weights=None)  # Jangan gunakan pretrained weights
        model.heads = torch.nn.Sequential(OrderedDict([
            ('head', torch.nn.Linear(in_features=768, out_features=3))  # Sesuaikan jumlah kelas
        ]))

        # Muat checkpoint dan terapkan state_dict ke model
        checkpoint = torch.load("best_model.pth", map_location=torch.device('cpu'))
        model.load_state_dict(checkpoint["model"])
        model.eval()
        st.success("Model loaded successfully!")
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

# Fungsi untuk preprocessing gambar
def preprocess_image(image):
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    return transform(image).unsqueeze(0)

# Load model
model = load_model()

# Hentikan aplikasi jika model gagal dimuat
if model is None:
    st.stop()

# Judul aplikasi
st.title("Dermatology Classification with ViT")
st.write("Upload an image of skin to classify it into one of the categories.")

# File uploader
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    try:
        # Tampilkan gambar yang diunggah
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Uploaded Image", use_container_width=True)

        # Preprocess gambar
        input_tensor = preprocess_image(image)

        # Prediksi dengan model
        with st.spinner("Classifying..."):
            outputs = model(input_tensor)
            _, predicted = torch.max(outputs, 1)

        # Label kelas
        class_names = ["acne", "bags", "redness"]  # Sesuaikan dengan dataset Anda
        st.write(f"Prediction: **{class_names[predicted.item()]}**")
    except Exception as e:
        st.error(f"Error processing image: {e}")
