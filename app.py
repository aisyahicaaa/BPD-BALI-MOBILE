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

/* Logo */
[data-testid="stImage"] {
    display: flex;
    justify-content: center;
    align-items: center;
    margin: 0 auto;
}

/* Text Area */
.stTextArea textarea {
    background: white !important;
    color: black !important;
    font-size: 16px !important;
    border-radius: 10px !important;
    padding: 12px !important;
}

/* Tombol */
.stButton > button {
    width: 100%;
    height: 50px;
    background: white !important;
    color: #0B6B3A !important;
    font-size: 18px !important;
    font-weight: bold !important;
    border-radius: 8px !important;
    border: none !important;
}

.stButton > button:hover {
    background: #F2F2F2 !important;
    color: #0B6B3A !important;
}

/* Warning dan hasil */
.stAlert {
    border-radius: 10px;
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
        "File model tidak ditemukan. "
        "Pastikan file .pkl berada dalam folder yang sama dengan app.py."
    )
    st.stop()



# ======================================
# HEADER LOGO
# ======================================

st.container(height=45, border=False)


kolom1, kolom2, kolom3 = st.columns([1.6, 1, 1.6])

with kolom2:
    st.image(
        "logo-bank-bpd-bali.png",
        use_container_width=True
    )


# Judul
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



# Subtitle
st.markdown("""
<p style="
text-align:center;
color:white;
font-size:18px;
margin-top:-3px;
margin-bottom:25px;">
Ulasan Aplikasi BPD Bali Mobile
</p>
""", unsafe_allow_html=True)



# ======================================
# INPUT ULASAN
# ======================================

st.markdown("""
<p style="
color:white;
font-size:18px;
font-weight:bold;
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



# ======================================
# PREDIKSI SENTIMEN
# ======================================

st.markdown(
    "<div style='margin-top:10px'></div>",
    unsafe_allow_html=True
)


if st.button("🔍 Analisis Sentimen"):

    if ulasan.strip() == "":
        st.warning("Masukkan ulasan terlebih dahulu.")

    else:

        # preprocessing
        hasil_preprocessing = preprocessing(ulasan)


        # vectorisasi
        hasil_vector = vectorizer.transform(
            [hasil_preprocessing]
        )


        # seleksi fitur chi-square
        hasil_vector = chi_selector.transform(
            hasil_vector
        )


        # prediksi
        hasil_prediksi = model.predict(
            hasil_vector
        )[0]


        st.markdown("<br>", unsafe_allow_html=True)


        if hasil_prediksi == 1:

            st.success(
                "😊 Sentimen Terdeteksi: POSITIF"
            )

        else:

            st.error(
                "😞 Sentimen Terdeteksi: NEGATIF"
            )


        # tampilkan preprocessing
        with st.expander(
            "Lihat Hasil Preprocessing"
        ):
            st.write(
                hasil_preprocessing
            )
