import streamlit as st
import pytesseract
import cv2
import numpy as np
import os
import urllib.request
from indic_transliteration import sanscript
from indic_transliteration.sanscript import transliterate
from weasyprint import HTML

# --- APP CONFIG & SESSION STATE ---
st.set_page_config(page_title="Odia Song OCR & Translator", layout="wide")

# Streamlit Memory (Session State)
if 'extracted_text' not in st.session_state:
    st.session_state.extracted_text = ""
if 'pdf_data' not in st.session_state:
    st.session_state.pdf_data = None

st.title("🎵 Odia Song OCR & Multilingual PDF Generator")
st.write("Upload an image, extract the text, fix any minor matra mistakes, and generate a formatted PDF in Odia, Hindi, and English.")

# --- 1. TESSERACT BEST MODEL SETUP ---
TESSDATA_DIR = os.path.abspath("tessdata")
os.makedirs(TESSDATA_DIR, exist_ok=True)
ORI_MODEL_PATH = os.path.join(TESSDATA_DIR, "ori.traineddata")

if not os.path.exists(ORI_MODEL_PATH):
    with st.spinner("Downloading High-Accuracy Odia AI Model (One-time setup)..."):
        url = "https://github.com/tesseract-ocr/tessdata_best/raw/main/ori.traineddata"
        urllib.request.urlretrieve(url, ORI_MODEL_PATH)

os.environ["TESSDATA_PREFIX"] = TESSDATA_DIR

# --- CORE FUNCTIONS ---
def clean_image(image_bytes):
    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    img_enlarged = cv2.resize(img, None, fx=3.0, fy=3.0, interpolation=cv2.INTER_CUBIC)
    gray = cv2.cvtColor(img_enlarged, cv2.COLOR_BGR2GRAY)
    denoised = cv2.fastNlMeansDenoising(gray, h=10)
    thresh = cv2.adaptiveThreshold(denoised, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 2)
    
    temp_path = "temp_image.png"
    cv2.imwrite(temp_path, thresh)
    return temp_path

def perform_ocr(image_path):
    custom_config = r'-c preserve_interword_spaces=1 --oem 1 --psm 6'
    text = pytesseract.image_to_string(image_path, lang='ori', config=custom_config)
    return text

def transliterate_text(odia_text):
    hindi_script = transliterate(odia_text, sanscript.ORIYA, sanscript.DEVANAGARI)
    eng_script = transliterate(odia_text, sanscript.ORIYA, sanscript.IAST)
    return hindi_script, eng_script

def create_pdf(odia_text, hindi_text, eng_text, output_filename="song.pdf"):
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="UTF-8">
        <link href="https://fonts.googleapis.com/css2?family=Noto+Sans+Devanagari&family=Noto+Sans+Oriya&family=Noto+Sans&display=swap" rel="stylesheet">
        <style>
            @page {{ size: A4; margin: 2cm; }}
            body {{ font-family: 'Noto Sans', sans-serif; }}
            h2 {{ color: #333; border-bottom: 2px solid #ddd; padding-bottom: 5px; margin-top: 30px; }}
            .text-block {{ white-space: pre-wrap; font-size: 14px; line-height: 1.5; }}
            .odia-text {{ font-family: 'Noto Sans Oriya', sans-serif; }}
            .hindi-text {{ font-family: 'Noto Sans Devanagari', sans-serif; }}
        </style>
    </head>
    <body>
        <h2>Original Odia Script:</h2>
        <div class="text-block odia-text">{odia_text}</div>
        
        <h2>Hindi (Devanagari) Script:</h2>
        <div class="text-block hindi-text">{hindi_text}</div>
        
        <h2>English (Latin) Script:</h2>
        <div class="text-block">{eng_text}</div>
    </body>
    </html>
    """
    HTML(string=html_content).write_pdf(output_filename)
    return output_filename

# --- UI WORKFLOW ---
uploaded_file = st.file_uploader("1. Upload Image (JPG/PNG)", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.image(uploaded_file, caption="Uploaded Image", use_container_width=True)
        if st.button("Scan & Extract Text", type="primary"):
            with st.spinner("Processing image with Advanced AI..."):
                temp_image_path = clean_image(uploaded_file.getvalue())
                st.session_state.extracted_text = perform_ocr(temp_image_path)
                os.remove(temp_image_path) 
                # Reset old PDF if user scans a new image
                st.session_state.pdf_data = None 
                st.success("Scan Complete! Please review the text on the right.")

    with col2:
        if st.session_state.extracted_text:
            st.info("💡 **Tip:** Edit the text below if you spot any missing matras before generating the PDF.")
            
            final_odia_text = st.text_area("2. Review & Edit Odia Text", value=st.session_state.extracted_text, height=400)
            
            # Step 1: Create PDF Button
            if st.button("3. Create Translation & PDF"):
                with st.spinner("Translating and formatting PDF..."):
                    hindi_text, eng_text = transliterate_text(final_odia_text)
                    pdf_file = create_pdf(final_odia_text, hindi_text, eng_text)
                    
                    # Save PDF to memory so it doesn't disappear
                    with open(pdf_file, "rb") as pdf:
                        st.session_state.pdf_data = pdf.read()
                    os.remove(pdf_file)
            
            # Step 2: Persistent Download Button
            if st.session_state.pdf_data is not None:
                st.success("✅ PDF is Ready!")
                st.download_button(
                    label="⬇️ Click Here to Download Final PDF",
                    data=st.session_state.pdf_data,
                    file_name="Translated_Song.pdf",
                    mime="application/pdf",
                    type="primary",
                    use_container_width=True
                )
