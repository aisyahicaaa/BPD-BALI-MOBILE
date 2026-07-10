import streamlit as st
import pickle
from preprocessing import preprocessing


# ======================================
# KONFIGURASI HALAMAN
# ======================================
st.set_page_config(
    page_title="Analisis Sentimen BPD Bali",
    page_icon="🏦",
    layout="centered"
)


# ======================================
# CSS TAMPILAN BPD BALI
# ======================================
st.markdown("""
<style>

.stApp {
    background-color: #0B6B3A;
}

.block-container {
    max-width: 680px;
    padding-top: 0px !important;
    padding-bottom: 40px;
}


[data-testid="stImage"] {
    display: flex;
    justify-content: center;
    align-items: center;
    margin: 0 auto;
}


.stTextArea textarea {
    background: white !important;
    color: black !important;
    font-size: 16px !important;
    border-radius: 10px !important;
    padding: 12px !important;
}


.stButton > button {

    width: 100%;
    height: 50px;
    background: white !important;
    color: #0B6B3A !important;
    font-size: 18px !important;
    font-weight: bold !important;
    border: none !important;
    border-radius: 8px !important;

}


.stButton > button:hover {

    background: #F2F2F2 !important;
    color: #0B6B3A !important;

}

</style>
""", unsafe_allow_html=True)



# ======================================
# LOAD MODEL
# ======================================
@st.cache_resource
def load_models():

    with open("vectorizer.pkl", "rb") as file:
        vectorizer = pickle.load(file)

    with open("chi_selector.pkl", "rb") as file:
        chi_selector = pickle.load(file)

    with open("model_naive_bayes.pkl", "rb") as file:
        model = pickle.load(file)

    return vectorizer, chi_selector, model



try:

    vectorizer, chi_selector, model = load_models()

except FileNotFoundError:

    st.error(
        "File model (.pkl) tidak ditemukan. "
        "Pastikan semua file berada dalam folder yang sama."
    )
    st.stop()



# ======================================
# HEADER LOGO
# ======================================

st.container(height=45, border=False)


kiri, tengah, kanan = st.columns([1.6, 1, 1.6])


with tengah:

    st.image(
        "logo-bank-bpd-bali.png",
        use_container_width=True
    )



# ======================================
# JUDUL
# ======================================

st.markdown("""
<h1 style="
text-align:center;
color:white;
font-size:38px;
font-weight:bold;
margin-top:-20px;
margin-bottom:0px;
line-height:1;">
Analisis Sentimen
</h1>
""", unsafe_allow_html=True)



st.markdown("""
<p style="
text-align:center;
color:rgba(255,255,255,0.9);
font-size:18px;
margin-top:-3px;
margin-bottom:20px;">
Ulasan Aplikasi BPD Bali Mobile
</p>
""", unsafe_allow_html=True)



# ======================================
# INPUT ULASAN
# ======================================

st.markdown("""
<p style="
font-size:18px;
font-weight:bold;
color:white;
margin-bottom:8px;">
Masukkan Ulasan
</p>
""", unsafe_allow_html=True)



ulasan = st.text_area(
    "",
    height=150,
    placeholder="Contoh: Aplikasi sangat membantu dan mudah digunakan.",
    label_visibility="collapsed"
)



st.markdown(
    "<div style='margin-top:10px'></div>",
    unsafe_allow_html=True
)



# ======================================
# PREDIKSI SENTIMEN
# ======================================

if st.button("🔍 Analisis Sentimen"):


    if ulasan.strip() == "":

        st.warning(
            "Masukkan ulasan terlebih dahulu."
        )


    else:


        # ==============================
        # PREPROCESSING
        # Cleaning
        # Case Folding
        # Tokenizing
        # Normalisasi
        # Stopword Removal
        # Stemming
        # ==============================

        hasil = preprocessing(ulasan)



        # ==============================
        # FEATURE EXTRACTION
        # ==============================

        vector = vectorizer.transform(
            [hasil]
        )



        # ==============================
        # CHI-SQUARE FEATURE SELECTION
        # ==============================

        vector = chi_selector.transform(
            vector
        )



        # ==============================
        # NAIVE BAYES PREDICTION
        # ==============================

        prediksi = model.predict(
            vector
        )[0]



        st.markdown(
            "<br>",
            unsafe_allow_html=True
        )



        if prediksi == 1:

            st.success(
                "😊 Sentimen Terdeteksi: POSITIF"
            )


        else:

            st.error(
                "😞 Sentimen Terdeteksi: NEGATIF"
            )



        # Menampilkan hasil preprocessing

        with st.expander(
            "Lihat Hasil Preprocessing"
        ):

            st.write(hasil)
