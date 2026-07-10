import streamlit as st
import pickle
from preprocessing import preprocess_text

# ============================
# LOAD MODEL
# ============================

with open("vectorizer.pkl", "rb") as file:
    vectorizer = pickle.load(file)

with open("chi_selector.pkl", "rb") as file:
    selector = pickle.load(file)

with open("model_naive_bayes.pkl", "rb") as file:
    model = pickle.load(file)

# ============================
# KONFIGURASI HALAMAN
# ============================

st.set_page_config(
    page_title="Analisis Sentimen BPD Bali Mobile",
    page_icon="💬",
    layout="centered"
)

# ============================
# CSS
# ============================

st.markdown("""
<style>

.main{
    background:#f7f7f7;
}

h1{
    color:#0a8f3d;
    text-align:center;
}

.stButton>button{
    background:#0a8f3d;
    color:white;
    width:100%;
    border-radius:8px;
    height:45px;
    font-size:17px;
}

textarea{
    border-radius:10px;
}

</style>
""", unsafe_allow_html=True)

# ============================
# JUDUL
# ============================

st.title("Analisis Sentimen")
st.subheader("Aplikasi BPD Bali Mobile")

st.write(
    "Masukkan ulasan pengguna kemudian tekan tombol **Analisis Sentimen**."
)

# ============================
# INPUT
# ============================

text = st.text_area(
    "Masukkan Ulasan",
    height=180,
    placeholder="Contoh : Aplikasi sangat membantu untuk transaksi."
)

# ============================
# PREDIKSI
# ============================

if st.button("Analisis Sentimen"):

    if text.strip()=="":

        st.warning("Masukkan ulasan terlebih dahulu.")

    else:

        clean = preprocess_text(text)

        vector = vectorizer.transform([clean])

        vector = selector.transform(vector)

        prediction = model.predict(vector)[0]

        probability = model.predict_proba(vector)

        if prediction==1:

            st.success("Sentimen : POSITIF 😊")

        else:

            st.error("Sentimen : NEGATIF 😔")

        st.write("### Hasil Preprocessing")

        st.code(clean)

        st.write("### Probabilitas")

        st.write(
            f"Negatif : {probability[0][0]*100:.2f}%"
        )

        st.write(
            f"Positif : {probability[0][1]*100:.2f}%"
        )