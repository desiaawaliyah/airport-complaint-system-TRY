import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from text_processor import preprocess_text
from model_trainer import load_model, predict_category
from utils import search_complaints
from sample_data import get_sample_data

# Set page configuration
st.set_page_config(
    page_title="Sistem Klasifikasi Komplain Penumpang Bandara",
    page_icon="✈️",
    layout="wide"
)

# Custom CSS untuk tema pink aesthetic
st.markdown("""
<style>
    .main-header {
        font-family: 'Arial', sans-serif;
        background: linear-gradient(to right, #FF85A2, #FFA7C4);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 2.5rem;
        font-weight: 800;
        margin-bottom: 1rem;
    }
    
    .subheader {
        color: #5E5E5E;
        font-size: 1.2rem;
        margin-bottom: 2rem;
    }
    
    .card {
        border: 1px solid #FFD1DC;
        border-radius: 15px;
        padding: 1.5rem;
        margin-bottom: 1rem;
        background-color: white;
        box-shadow: 0 4px 10px rgba(255, 133, 162, 0.2);
    }
    
    .category-tag {
        background-color: #FF85A2;
        color: white;
        padding: 0.2rem 0.5rem;
        border-radius: 20px;
        font-size: 0.8rem;
        font-weight: bold;
    }
    
    .aesthetic-footer {
        text-align: center;
        margin-top: 2rem;
        padding-top: 1rem;
        border-top: 1px solid #FFD1DC;
        color: #888;
        font-size: 0.8rem;
    }
    
    /* Cute aesthetic icons */
    .aesthetic-icon {
        font-size: 1.5rem;
        margin-right: 0.5rem;
        color: #FF85A2;
    }
    
    /* Rounded containers */
    .stButton>button {
        border-radius: 20px;
        background-color: #FF85A2;
        color: white;
        border: none;
        padding: 0.5rem 1rem;
    }
    
    .stButton>button:hover {
        background-color: #FFA7C4;
    }
    
    /* Softer inputs */
    .stTextInput>div>div>input {
        border-radius: 20px;
        border-color: #FFD1DC;
    }
    
    /* Prettier dividers */
    hr {
        border-top: 1px dashed #FFD1DC;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state variables if they don't exist
if 'complaints_df' not in st.session_state:
    st.session_state.complaints_df = pd.DataFrame(columns=['text', 'category', 'confidence', 'date'])

if 'model' not in st.session_state or 'vectorizer' not in st.session_state:
    with st.spinner('Loading model... This might take a minute.'):
        st.session_state.model, st.session_state.vectorizer, st.session_state.label_encoder = load_model()

# Page title and description
st.markdown('<h1 class="main-header">✈️ Sistem Klasifikasi Komplain Penumpang Bandara</h1>', unsafe_allow_html=True)
st.markdown('<p class="subheader">Analisis dan klasifikasi komplain penumpang bandara dengan Natural Language Processing</p>', unsafe_allow_html=True)

# Tambahkan gambar bandara di header (tapi dengan filter pink)
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    st.image("https://img.freepik.com/free-vector/airport-departure-lounge-concept-illustration_114360-7532.jpg", 
             use_container_width=True, 
             caption="Terminal Bandara - Pink Edition")

# Informasi bandara dengan tema pink
st.markdown("""
<div class="card">
    <span class="aesthetic-icon">🛫</span> Sistem ini dirancang khusus untuk mengelola dan menganalisis komplain penumpang di bandara.
    <br>
    <span class="aesthetic-icon">🧳</span> Mengidentifikasi kategori komplain terkait fasilitas, layanan, keamanan, dan lainnya.
    <br>
    <span class="aesthetic-icon">📊</span> Menyediakan visualisasi untuk memahami pola dan tren komplain.
</div>
""", unsafe_allow_html=True)

# Sidebar for inputs and filters
with st.sidebar:
    st.markdown('<h3 style="color:#FF85A2;"><span style="font-size:1.5rem;">✈️</span> Masukkan Komplain Bandara</h3>', unsafe_allow_html=True)
    
    # Input options
    input_option = st.radio("Pilih metode input:", ["Masukkan Teks", "Unggah File"])
    
    if input_option == "Masukkan Teks":
        complaint_text = st.text_area("Ketik komplain di sini:", height=150)
        submit_button = st.button("Proses Komplain")
        
        if submit_button and complaint_text:
            # Preprocess the text
            processed_text = preprocess_text(complaint_text)
            
            # Predict category
            category, confidence = predict_category(
                processed_text, 
                st.session_state.model, 
                st.session_state.vectorizer,
                st.session_state.label_encoder
            )
            
            # Add to dataframe
            new_complaint = pd.DataFrame({
                'text': [complaint_text],
                'category': [category],
                'confidence': [confidence],
                'date': [pd.Timestamp.now()]
            })
            st.session_state.complaints_df = pd.concat([st.session_state.complaints_df, new_complaint], ignore_index=True)
            st.success(f"Komplain berhasil diklasifikasikan sebagai '{category}' dengan tingkat kepercayaan {confidence:.2f}%")
    
    else:  # Upload File
        uploaded_file = st.file_uploader("Pilih file CSV atau TXT:", type=['csv', 'txt'])
        if uploaded_file is not None:
            try:
                if uploaded_file.name.endswith('.csv'):
                    df = pd.read_csv(uploaded_file)
                    if 'text' not in df.columns:
                        st.error("File CSV harus memiliki kolom 'text' yang berisi komplain.")
                    else:
                        process_button = st.button("Proses Komplain")
                        if process_button:
                            with st.spinner("Memproses komplain..."):
                                # Process each complaint
                                results = []
                                for idx, row in df.iterrows():
                                    text = row['text']
                                    processed_text = preprocess_text(text)
                                    category, confidence = predict_category(
                                        processed_text, 
                                        st.session_state.model, 
                                        st.session_state.vectorizer,
                                        st.session_state.label_encoder
                                    )
                                    results.append({
                                        'text': text,
                                        'category': category,
                                        'confidence': confidence,
                                        'date': pd.Timestamp.now()
                                    })
                                
                                # Add to session state
                                new_complaints = pd.DataFrame(results)
                                st.session_state.complaints_df = pd.concat([st.session_state.complaints_df, new_complaints], ignore_index=True)
                                st.success(f"Berhasil memproses {len(new_complaints)} komplain")
                
                elif uploaded_file.name.endswith('.txt'):
                    content = uploaded_file.read().decode()
                    complaints = [c.strip() for c in content.split('\n') if c.strip()]
                    
                    process_button = st.button("Proses Komplain")
                    if process_button:
                        with st.spinner("Memproses komplain..."):
                            # Process each complaint
                            results = []
                            for text in complaints:
                                processed_text = preprocess_text(text)
                                category, confidence = predict_category(
                                    processed_text, 
                                    st.session_state.model, 
                                    st.session_state.vectorizer,
                                    st.session_state.label_encoder
                                )
                                results.append({
                                    'text': text,
                                    'category': category,
                                    'confidence': confidence,
                                    'date': pd.Timestamp.now()
                                })
                            
                            # Add to session state
                            new_complaints = pd.DataFrame(results)
                            st.session_state.complaints_df = pd.concat([st.session_state.complaints_df, new_complaints], ignore_index=True)
                            st.success(f"Berhasil memproses {len(new_complaints)} komplain")
            
            except Exception as e:
                st.error(f"Error memproses file: {str(e)}")
    
    # Load sample data option
    st.divider()
    if st.button("Muat Data Sampel"):
        with st.spinner("Memuat data sampel..."):
            sample_data = get_sample_data()
            
            results = []
            for text in sample_data:
                processed_text = preprocess_text(text)
                category, confidence = predict_category(
                    processed_text, 
                    st.session_state.model, 
                    st.session_state.vectorizer,
                    st.session_state.label_encoder
                )
                results.append({
                    'text': text,
                    'category': category,
                    'confidence': confidence,
                    'date': pd.Timestamp.now()
                })
            
            st.session_state.complaints_df = pd.DataFrame(results)
            st.success("Data sampel berhasil dimuat!")
    
    # Filter section
    st.divider()
    st.header("Filter")
    
    # Only show filters if we have data
    if not st.session_state.complaints_df.empty:
        # Category filter
        all_categories = st.session_state.complaints_df['category'].unique().tolist()
        selected_categories = st.multiselect(
            "Pilih kategori:",
            options=all_categories,
            default=all_categories
        )
        
        # Confidence filter
        min_confidence = st.slider(
            "Tingkat kepercayaan minimum:",
            min_value=0.0,
            max_value=100.0,
            value=0.0,
            step=5.0
        )
        
        # Apply filters
        filtered_df = st.session_state.complaints_df[
            (st.session_state.complaints_df['category'].isin(selected_categories)) &
            (st.session_state.complaints_df['confidence'] >= min_confidence)
        ]
    else:
        filtered_df = st.session_state.complaints_df.copy()

# Main content area
tab1, tab2, tab3, tab4 = st.tabs(["🛫 Dashboard", "🧳 Daftar Komplain", "🔎 Pencarian", "📋 Info Laporan"])

# Dashboard tab
with tab1:
    if not filtered_df.empty:
        st.markdown('<h2 style="color:#FF85A2;"><span style="font-size:1.5rem;">✈️</span> Dashboard Komplain Bandara</h2>', unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Kategori Komplain chart
            category_counts = filtered_df['category'].value_counts().reset_index()
            category_counts.columns = ['Category', 'Count']
            
            fig = px.pie(
                category_counts, 
                values='Count', 
                names='Category',
                title='Distribusi Kategori Komplain',
                hole=0.4,
                color_discrete_sequence=['#FF85A2', '#FFA7C4', '#FFD1DC', '#FFDBE7', '#FFE6EF', '#FFF0F5']
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Confidence Distribution chart
            fig = px.histogram(
                filtered_df,
                x='confidence',
                nbins=20,
                title='Distribusi Tingkat Kepercayaan',
                labels={'confidence': 'Tingkat Kepercayaan (%)', 'count': 'Jumlah Komplain'},
                color_discrete_sequence=['#FF85A2']
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # Category-wise confidence
        st.subheader("Rata-rata Tingkat Kepercayaan per Kategori")
        avg_confidence = filtered_df.groupby('category')['confidence'].mean().reset_index()
        avg_confidence.columns = ['Category', 'Average Confidence']
        
        fig = px.bar(
            avg_confidence,
            x='Category',
            y='Average Confidence',
            title='Rata-rata Tingkat Kepercayaan per Kategori',
            labels={'Category': 'Kategori', 'Average Confidence': 'Rata-rata Tingkat Kepercayaan (%)'},
            color='Average Confidence',
            color_continuous_scale=[[0, '#FFD1DC'], [0.5, '#FFA7C4'], [1, '#FF85A2']]
        )
        st.plotly_chart(fig, use_container_width=True)
        
    else:
        st.info("Belum ada data komplain untuk ditampilkan. Silakan masukkan komplain atau muat data sampel terlebih dahulu.")

# List of complaints tab
with tab2:
    if not filtered_df.empty:
        st.markdown('<h2 style="color:#FF85A2;"><span style="font-size:1.5rem;">🧳</span> Daftar Komplain Bandara</h2>', unsafe_allow_html=True)
        
        # Add export button
        from utils import export_to_csv
        
        if st.button("Ekspor ke CSV"):
            csv_path = export_to_csv(filtered_df)
            if csv_path:
                with open(csv_path, 'rb') as file:
                    csv_contents = file.read()
                    st.download_button(
                        label="Unduh File CSV",
                        data=csv_contents,
                        file_name="komplain_terklasifikasi.csv",
                        mime="text/csv"
                    )
                    st.success(f"Data berhasil diekspor ke {csv_path}")
            else:
                st.error("Gagal mengekspor data")
        
        # Display data with formatting
        for i, row in filtered_df.iterrows():
            with st.container():
                col1, col2 = st.columns([3, 1])
                
                with col1:
                    st.markdown(f"**Komplain:** {row['text']}")
                
                with col2:
                    st.markdown(f"**Kategori:** {row['category']}")
                    st.markdown(f"**Kepercayaan:** {row['confidence']:.2f}%")
                
                st.divider()
    else:
        st.info("Belum ada data komplain untuk ditampilkan. Silakan masukkan komplain atau muat data sampel terlebih dahulu.")

# Search tab
with tab3:
    st.markdown('<h2 style="color:#FF85A2;"><span style="font-size:1.5rem;">🔎</span> Pencarian Komplain Bandara</h2>', unsafe_allow_html=True)
    
    search_query = st.text_input("Cari komplain berdasarkan kata kunci:")
    
    if search_query and not st.session_state.complaints_df.empty:
        search_results = search_complaints(st.session_state.complaints_df, search_query)
        
        if not search_results.empty:
            st.subheader(f"Hasil Pencarian ({len(search_results)} hasil)")
            
            for i, row in search_results.iterrows():
                with st.container():
                    col1, col2 = st.columns([3, 1])
                    
                    with col1:
                        st.markdown(f"**Komplain:** {row['text']}")
                    
                    with col2:
                        st.markdown(f"**Kategori:** {row['category']}")
                        st.markdown(f"**Kepercayaan:** {row['confidence']:.2f}%")
                    
                    st.divider()
        else:
            st.info("Tidak ada hasil yang ditemukan untuk kata kunci tersebut.")
    elif search_query:
        st.info("Belum ada data komplain untuk dicari. Silakan masukkan komplain atau muat data sampel terlebih dahulu.")

# Info Laporan tab
with tab4:
    st.markdown('<h2 style="color:#FF85A2;"><span style="font-size:1.5rem;">📋</span> Informasi untuk Laporan Bandara</h2>', unsafe_allow_html=True)
    
    st.subheader("Dokumentasi Sistem")
    
    # Overview
    st.markdown("""
    ## Sistem Klasifikasi Komplain Penumpang Bandara
    
    Sistem ini adalah aplikasi berbasis web yang menggunakan Natural Language Processing (NLP) untuk
    mengklasifikasikan dan menganalisis komplain penumpang bandara dalam bahasa Indonesia. 
    
    ### Fitur Utama
    
    1. **Klasifikasi Komplain** - Mengkategorikan komplain ke dalam 6 kategori:
       - Layanan
       - Fasilitas
       - Teknis
       - Keamanan
       - Harga
       - Kenyamanan
       
    2. **Analisis Sentimen** - Mengidentifikasi isu-isu utama dan pola komplain
    
    3. **Visualisasi Data** - Menampilkan distribusi kategori dan tingkat kepercayaan
    
    4. **Pencarian Komplain** - Mencari komplain berdasarkan kata kunci
    
    ### Teknologi yang Digunakan
    
    - **Frontend**: Streamlit
    - **Backend**: Python
    - **NLP**: Custom Text Processor
    - **Machine Learning**: Scikit-learn (TF-IDF + Naive Bayes)
    - **Visualisasi**: Plotly
    """)
    
    # Technical details
    st.subheader("Detail Teknis untuk Laporan")
    
    with st.expander("Arsitektur Sistem"):
        st.markdown("""
        ### Arsitektur Sistem
        
        Sistem terdiri dari beberapa komponen utama:
        
        1. **Text Processor** - Melakukan preprocessing teks:
           - Konversi ke lowercase
           - Penghapusan karakter khusus
           - Tokenisasi
           - Penghapusan stopwords
           - Stemming
        
        2. **Model Klasifikasi** - Menggunakan TF-IDF dan Naive Bayes
        
        3. **Antarmuka Web** - Menggunakan Streamlit untuk interaksi pengguna
        """)
    
    with st.expander("Model Machine Learning"):
        st.markdown("""
        ### Model Machine Learning
        
        Model klasifikasi menggunakan pendekatan berikut:
        
        - **Vectorizer**: TF-IDF (Term Frequency-Inverse Document Frequency)
        - **Classifier**: Multinomial Naive Bayes
        - **Training Data**: 60 sampel komplain dengan label
        - **Evaluasi**: Train-test split (80/20)
        - **Akurasi Model**: ~33% (versi awal)
        
        Model ini memiliki akurasi yang masih dapat ditingkatkan melalui:
        - Penambahan data training
        - Optimisasi hyperparameter
        - Penggunaan model yang lebih kompleks seperti LSTM atau BERT
        """)
    
    with st.expander("Referensi dan Sumber"):
        st.markdown("""
        ### Referensi dan Sumber
        
        - Scikit-learn Documentation: [https://scikit-learn.org/](https://scikit-learn.org/)
        - Streamlit Documentation: [https://docs.streamlit.io/](https://docs.streamlit.io/)
        - Plotly Documentation: [https://plotly.com/python/](https://plotly.com/python/)
        - Natural Language Processing with Python by Bird, Klein, and Loper
        """)
    
    # Code snippets
    st.subheader("Contoh Kode untuk Laporan")
    
    with st.expander("Text Preprocessing"):
        st.code('''
# Preprocessing teks
def preprocess_text(text):
    # Convert to lowercase
    text = text.lower()
    
    # Remove URLs and emails
    text = re.sub(r'https?://\\S+|www\\.\\S+', '', text)
    text = re.sub(r'\\S+@\\S+', '', text)
    
    # Remove punctuation and special characters
    text = re.sub(r'[^\\w\\s]', '', text)
    
    # Tokenize
    tokens = simple_word_tokenize(text)
    
    # Remove stopwords
    tokens = [token for token in tokens if token not in STOPWORDS]
    
    # Stemming
    tokens = [simple_stem(token) for token in tokens]
    
    return ' '.join(tokens)
''', language='python')
    
    with st.expander("Model Training"):
        st.code('''
# Training model klasifikasi
def train_model():
    # Get training data
    train_data = get_sample_data_with_labels()
    df = pd.DataFrame(train_data, columns=['text', 'category'])
    
    # Preprocess text
    df['processed_text'] = df['text'].apply(preprocess_text)
    
    # Encode categories
    label_encoder = LabelEncoder()
    df['category_encoded'] = label_encoder.fit_transform(df['category'])
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        df['processed_text'], 
        df['category_encoded'],
        test_size=0.2,
        random_state=42
    )
    
    # Create TF-IDF vectorizer
    vectorizer = TfidfVectorizer(max_features=5000)
    X_train_vectorized = vectorizer.fit_transform(X_train)
    
    # Train Naive Bayes model
    model = MultinomialNB()
    model.fit(X_train_vectorized, y_train)
    
    # Evaluate
    X_test_vectorized = vectorizer.transform(X_test)
    accuracy = model.score(X_test_vectorized, y_test)
    
    return model, vectorizer, label_encoder
''', language='python')

    # Provide installation instructions
    st.subheader("Instalasi dan Pengembangan")
    
    with st.expander("Instalasi"):
        st.markdown("""
        ### Instalasi Lokal
        
        1. **Clone repository**:
        ```
        git clone https://your-repository-url.git
        cd passenger-complaint-system
        ```
        
        2. **Instal dependency**:
        ```
        pip install streamlit pandas numpy plotly scikit-learn
        ```
        
        3. **Jalankan aplikasi**:
        ```
        streamlit run app.py
        ```
        """)
    
    # Links to download all code files
    st.subheader("Download Kode Sumber")
    st.markdown("""
    File kode sumber dapat diunduh dari Replit atau GitHub. Pastikan untuk menyertakan file-file berikut dalam laporan Anda:
    
    - app.py - Aplikasi utama
    - text_processor.py - Pemrosesan teks
    - model_trainer.py - Pelatihan model
    - utils.py - Fungsi utilitas
    - sample_data.py - Data sampel
    - .streamlit/config.toml - Konfigurasi Streamlit
    """)
    
# Add footer with pink airport theme
st.markdown("""
<div class="aesthetic-footer">
    <p>Sistem Klasifikasi Komplain Penumpang Bandara © 2025</p>
    <p style="font-size:0.8rem; color:#666;">
        <span style="color:#FF85A2; margin:0 5px;">✈️</span> Analisis Sentimen 
        <span style="color:#FF85A2; margin:0 5px;">🛫</span> Klasifikasi Komplain
        <span style="color:#FF85A2; margin:0 5px;">🧳</span> Visualisasi Data
    </p>
</div>
""", unsafe_allow_html=True)
