import streamlit as st
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer

# Fungsi untuk memuat model dan vectorizer
def load_model(file_path):
    try:
        with open(file_path, 'rb') as file:
            return pickle.load(file)
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

# Fungsi utama aplikasi
def main():
    # Judul aplikasi
    st.title("Analisis Sentimen")

    # Input teks dari pengguna
    user_input = st.text_input("Masukkan teks untuk analisis sentimen:")

    # Tombol prediksi
    if st.button("Prediksi Sentimen"):
        if not user_input:
            st.warning("Harap masukkan teks sebelum melakukan prediksi.")
        else:
            # Muat model dan vectorizer
            model = load_model('rf_model.pkl')  # Ganti dengan path model Anda
            vectorizer = load_model('vectorizer.pkl')  # Ganti dengan path vectorizer Anda
            
            if model and vectorizer:
                # Transformasi teks menggunakan vectorizer
                input_vector = vectorizer.transform([user_input])
                
                # Prediksi sentimen
                sentiment = model.predict(input_vector)[0]

                # Menampilkan hasil prediksi
                if sentiment == 1:
                    st.success("Hasil Sentimen: Positif 😊")
                elif sentiment == 0:
                    st.warning("Hasil Sentimen: Negatif 😢")
                else:
                    st.info("Hasil Sentimen: Netral 😐")

if __name__ == '__main__':
    main()
