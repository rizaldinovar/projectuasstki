import streamlit as st
import torch
from torchvision.models import vit_b_16
from torchvision import transforms
from PIL import Image
from collections import OrderedDict
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics.pairwise import cosine_similarity
import re
from sentence_transformers import SentenceTransformer  # Menggunakan Sentence-BERT

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

# Fungsi untuk membersihkan teks
def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)
    return text

# Memuat model Sentence-BERT
@st.cache_resource
def load_sentence_bert_model():
    model = SentenceTransformer('all-MiniLM-L6-v2')  # Memuat model Sentence-BERT
    return model

# Memuat model dan Sentence-BERT
model = load_model()
sentence_bert_model = load_sentence_bert_model()

# Hentikan aplikasi jika model gagal dimuat
if model is None or sentence_bert_model is None:
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

# Membaca dataset yang sudah dibersihkan
file_path = 'C:/Users/rizal/Videos/stkif/dataset_cleaned.csv'
data = pd.read_csv(file_path)

# Terapkan fungsi preprocessing pada dataset
data['deskripsi_produk'] = data['deskripsi_produk'].apply(clean_text)
data['masalah_kulit_yang_ditangani'] = data['masalah_kulit_yang_ditangani'].apply(clean_text)

# Menambahkan kategori produk
# Pastikan kolom kategori_produk ada di dataset Anda
data['kategori_produk'] = data['kategori_produk'].apply(clean_text)

# KNN untuk Content-Based Filtering
label_encoder = LabelEncoder()
data['jenis_kulit_yang_cocok_encoded'] = label_encoder.fit_transform(data['jenis_kulit_yang_cocok'])

knn = NearestNeighbors(n_neighbors=5, metric='cosine')
X = data[['jenis_kulit_yang_cocok_encoded']].values  # Fitur jenis kulit yang cocok
knn.fit(X)

# Fungsi untuk mencari kategori berdasarkan query
def get_category_from_query(query, label_encoder):
    try:
        # Mencoba untuk mengubah query menjadi kategori encoded
        query_category_encoded = label_encoder.transform([query.lower()])[0]
    except ValueError:
        # Jika label tidak ditemukan, pilih kategori default atau yang paling mendekati
        st.warning("Kategori tidak ditemukan. Menampilkan rekomendasi skincare berdasarkan masalah kulit default.")
        query_category_encoded = 0  # Misalnya memilih kategori pertama sebagai default
    return query_category_encoded

# Fungsi untuk mendapatkan vektor dari Sentence-BERT
def get_sentence_bert_vector(text, model):
    return model.encode(text)  # Mendapatkan vektor kalimat menggunakan Sentence-BERT

# Query pencarian
query = st.text_input("Masukkan masalah kulit Anda untuk mencari produk skincare:", skin_condition if uploaded_file else "")

if st.button('Cari'):
    if query:
        # Preprocessing query untuk mendapatkan vektor dari Sentence-BERT
        query_vector = get_sentence_bert_vector(query.lower(), sentence_bert_model)

        # Cosine Similarity antara query dan deskripsi produk
        cosine_similarities = []
        for i, desc in enumerate(data['deskripsi_produk']):
            product_vector = get_sentence_bert_vector(desc, sentence_bert_model)
            similarity = cosine_similarity([query_vector], [product_vector])[0][0]
            cosine_similarities.append(similarity)

        # Menampilkan hasil berdasarkan IR
        top_indices_ir = np.argsort(cosine_similarities)[-5:][::-1]
        st.subheader("Hasil Pencarian Skincare Berdasarkan Masalah Kulit Anda")
        for idx in top_indices_ir:
            st.write(f"Produk: {data['nama_produk'].iloc[idx]}")
            st.write(f"Brand: {data['nama_brand'].iloc[idx]}")
            st.write(f"Deskripsi: {data['deskripsi_produk'].iloc[idx]}")
            st.write(f"Masalah Kulit yang Ditangani: {data['masalah_kulit_yang_ditangani'].iloc[idx]}")
            st.write(f"Kategori Produk: {data['kategori_produk'].iloc[idx]}")
            st.write(f"Harga: {data['harga_produk'].iloc[idx]}")
            st.write(f"Link Pembelian: {data['link_pembelian'].iloc[idx]}")
            st.write("---")