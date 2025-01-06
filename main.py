import pandas as pd # type: ignore
import numpy as np # type: ignore
import streamlit as st # type: ignore
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

# Membaca dataset yang sudah dibersihkan
file_path = 'C:/Users/rizal/Videos/stkif/dataset_cleaned.csv'
data = pd.read_csv(file_path)

# Preprocessing: Menghapus nilai kosong dan karakter non-alfabet
def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)
    return text

data['deskripsi_produk'] = data['deskripsi_produk'].apply(clean_text)
data['masalah_kulit_yang_ditangani'] = data['masalah_kulit_yang_ditangani'].apply(clean_text)

# TF-IDF Vectorization untuk Deskripsi Produk (untuk IR)
tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(data['deskripsi_produk'])

# KNN untuk Content-Based Filtering
# Menggunakan fitur 'jenis_kulit_yang_cocok' dan 'kategori_produk' untuk KNN
label_encoder = LabelEncoder()
data['jenis_kulit_yang_cocok_encoded'] = label_encoder.fit_transform(data['jenis_kulit_yang_cocok'])

# Model KNN untuk Content-Based Filtering
knn = NearestNeighbors(n_neighbors=5, metric='cosine')
X = data[['jenis_kulit_yang_cocok_encoded']].values  # Fitur jenis kulit yang cocok
knn.fit(X)

# NLP: Named Entity Recognition (NER) untuk masalah kulit
# Menggunakan pencocokan kata kunci untuk masalah kulit
def get_skin_problem_from_text(text):
    skin_problems = ['acne', 'wrinkles', 'dark spots', 'pore care', 'redness', 'dryness', 'oiliness']
    text = text.lower()
    found_problems = [problem for problem in skin_problems if problem in text]
    return found_problems if found_problems else ['Unknown']

data['detected_skin_problems'] = data['masalah_kulit_yang_ditangani'].apply(get_skin_problem_from_text)

# Streamlit UI untuk aplikasi pencarian
def app():
    st.title('Search Engine Skincare Recommendation System Based on Skin Problems')

    # Input Query: Masukkan masalah kulit pengguna
    query = st.text_input("Masukkan masalah kulit yang Anda alami:", "")

    # Button untuk men-trigger pencarian
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
                st.write(f"Harga: Rp {data['harga_produk'].iloc[idx]}")
                st.write(f"Link Pembelian: {data['link_pembelian'].iloc[idx]}")
                st.write("---")

    # Input Jenis Kulit dan Kategori Produk untuk CBF
    skin_type = st.selectbox("Pilih jenis kulit Anda:", ['Normal', 'Dry', 'Oily', 'Sensitive', 'Combination'])
    if skin_type:
        # Menyaring produk berdasarkan jenis kulit
        filtered_data = data[data['jenis_kulit_yang_cocok'].str.contains(skin_type, case=False)]
        
        st.subheader(f"Rekomendasi Skincare untuk {skin_type} Skin")
        for _, row in filtered_data.iterrows():
            st.write(f"Produk: {row['nama_produk']}")
            st.write(f"Brand: {row['nama_brand']}")
            st.write(f"Deskripsi: {row['deskripsi_produk']}")
            st.write(f"Harga: Rp {row['harga_produk']}")
            st.write(f"Link Pembelian: {row['link_pembelian']}")
            st.write("---")

    # Input untuk Identifikasi Masalah Kulit dengan NLP
    st.subheader("NLP: Menyaring Berdasarkan Masalah Kulit")
    skin_problem_query = st.text_input("Masukkan masalah kulit yang ingin Anda cari (misal: acne, dryness, etc.):", "")

    if skin_problem_query:
        skin_problem_matches = data[data['detected_skin_problems'].apply(lambda x: any(problem in x for problem in get_skin_problem_from_text(skin_problem_query)))]

        st.subheader(f"Rekomendasi untuk {skin_problem_query.capitalize()}")
        for _, row in skin_problem_matches.iterrows():
            st.write(f"Produk: {row['nama_produk']}")
            st.write(f"Brand: {row['nama_brand']}")
            st.write(f"Deskripsi: {row['deskripsi_produk']}")
            st.write(f"Harga: Rp {row['harga_produk']}")
            st.write(f"Link Pembelian: {row['link_pembelian'].iloc[0]}")
            st.write("---")

if __name__ == "__main__":
    app()