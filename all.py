import streamlit as st # type: ignore
import torch # type: ignore
from torchvision.models import vit_b_16 # type: ignore
from torchvision import transforms # type: ignore
from PIL import Image # type: ignore
from collections import OrderedDict
import pandas as pd # type: ignore
import numpy as np # type: ignore
from sklearn.feature_extraction.text import TfidfVectorizer # type: ignore
from sklearn.metrics.pairwise import cosine_similarity # type: ignore
from sklearn.neighbors import NearestNeighbors # type: ignore
from sklearn.preprocessing import LabelEncoder # type: ignore
import nltk # type: ignore
from nltk.tokenize import word_tokenize # type: ignore
from nltk.corpus import stopwords # type: ignore
import re

# Download NLTK resources
nltk.download('punkt')
nltk.download('stopwords')

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
st.title("Dermatology Classification and Skincare Recommendation")
st.write("Upload an image of skin to classify it and find suitable skincare products.")

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
        skin_condition = class_names[predicted.item()]
        st.write(f"Prediction: **{skin_condition}**")

    except Exception as e:
        st.error(f"Error processing image: {e}")

# Setelah prediksi, lanjut ke pencarian dan rekomendasi skincare

# Membaca dataset yang sudah dibersihkan
file_path = 'C:/Users/rizal/Videos/stkif/dataset_cleaned.csv'
data = pd.read_csv(file_path)

# Preprocessing: Menghapus nilai kosong dan karakter non-alfabet
def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)
    return text

# Fungsi untuk membersihkan kolom harga
def clean_price(price):
    # Hapus duplikasi "Rp" dan karakter non-digit
    clean_value = re.sub(r'[^0-9]', '', price)  # Ambil hanya angka
    # Format dengan pemisah ribuan
    formatted_price = f"Rp {int(clean_value):,}".replace(",", ".")
    return formatted_price

# Terapkan fungsi preprocessing pada dataset
data['deskripsi_produk'] = data['deskripsi_produk'].apply(clean_text)
data['masalah_kulit_yang_ditangani'] = data['masalah_kulit_yang_ditangani'].apply(clean_text)
data['harga_produk'] = data['harga_produk'].apply(clean_price)  # Perbaikan harga

# TF-IDF Vectorization untuk Deskripsi Produk (untuk IR)
tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(data['deskripsi_produk'])

# KNN untuk Content-Based Filtering
label_encoder = LabelEncoder()
data['jenis_kulit_yang_cocok_encoded'] = label_encoder.fit_transform(data['jenis_kulit_yang_cocok'])

knn = NearestNeighbors(n_neighbors=5, metric='cosine')
X = data[['jenis_kulit_yang_cocok_encoded']].values  # Fitur jenis kulit yang cocok
knn.fit(X)

# Query pencarian
query = st.text_input("Masukkan masalah kulit Anda untuk mencari produk skincare:", skin_condition if uploaded_file else "")

if st.button('Cari'):
    if query:
        # Proses Query untuk IR (TF-IDF + Cosine Similarity)
        query_tfidf = tfidf.transform([query.lower()])
        cosine_similarities = cosine_similarity(query_tfidf, tfidf_matrix).flatten()

        # Menampilkan hasil berdasarkan IR
        top_indices_ir = cosine_similarities.argsort()[-5:][::-1]
        st.subheader("Hasil Pencarian Skincare Berdasarkan Masalah Kulit Anda")
        for idx in top_indices_ir:
            st.write(f"Produk: {data['nama_produk'].iloc[idx]}")
            st.write(f"Brand: {data['nama_brand'].iloc[idx]}")
            st.write(f"Deskripsi: {data['deskripsi_produk'].iloc[idx]}")
            st.write(f"Masalah Kulit yang Ditangani: {data['masalah_kulit_yang_ditangani'].iloc[idx]}")
            st.write(f"Harga: {data['harga_produk'].iloc[idx]}")  # Harga terformat
            st.write(f"Link Pembelian: {data['link_pembelian'].iloc[idx]}")
            st.write("---")