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

if 'extracted_text' not in st.session_state:
    st.session_state.extracted_text = ""

st.title("🎵 Odia Song OCR & Multilingual PDF Generator")
st.write("Upload an image, extract the text, fix any minor matra mistakes, and generate a formatted PDF in Odia, Hindi, and English.")

# --- 1. TESSERACT BEST MODEL SETUP ---
# Yeh code automatically Google ka sabse best Odia AI model download karega
TESSDATA_DIR = os.path.abspath("tessdata")
os.makedirs(TESSDATA_DIR, exist_ok=True)
ORI_MODEL_PATH = os.path.join(TESSDATA_DIR, "ori.traineddata")

if not os.path.exists(ORI_MODEL_PATH):
    with st.spinner("Downloading High-Accuracy Odia AI Model (One-time setup)..."):
        url = "https://github.com/tesseract-ocr/tessdata_best/raw/main/ori.traineddata"
        urllib.request.urlretrieve(url, ORI_MODEL_PATH)

# Force pytesseract to use our newly downloaded best model
os.environ["TESSDATA_PREFIX"] = TESSDATA_DIR

# --- CORE FUNCTIONS ---
def clean_image(image_bytes):
    """Pro-level image processing for complex Indic scripts."""
    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    # 1. Extreme Upscaling (3x) to make tiny matras visible
    img_enlarged = cv2.resize(img, None, fx=3.0, fy=3.0, interpolation=cv2.INTER_CUBIC)
    
    # 2. Convert to Grayscale
    gray = cv2.cvtColor(img_enlarged, cv2.COLOR_BGR2GRAY)
    
    # 3. Denoising to remove background grain/shadows
    denoised = cv2.fastNlMeansDenoising(gray, h=10)
    
    # 4. Adaptive Gaussian Thresholding 
    # (Forces text to be pure black and bg pure white, adjusting for lighting locally)
    thresh = cv2.adaptiveThreshold(denoised, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 2)
    
    temp_path = "temp_image.png"
    cv2.imwrite(temp_path, thresh)
    return temp_path

def perform_ocr(image_path):
    # --oem 1 forces deep learning. --psm 6 preserves layout columns.
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
                st.success("Scan Complete! Please review the text on the right.")

    with col2:
        if st.session_state.extracted_text:
            st.info("💡 **Tip:** Even with advanced AI, OCR can make a 1% error on rare matras. Feel free to edit the text below before generating the PDF.")
            
            final_odia_text = st.text_area("2. Review & Edit Odia Text", value=st.session_state.extracted_text, height=400)
            
            if st.button("3. Translate & Download PDF", type="primary"):
                with st.spinner("Translating and formatting PDF..."):
                    hindi_text, eng_text = transliterate_text(final_odia_text)
                    
                    pdf_file = create_pdf(final_odia_text, hindi_text, eng_text)
                    
                    with open(pdf_file, "rb") as pdf:
                        st.download_button(
                            label="⬇️ Download Final PDF",
                            data=pdf,
                            file_name="Translated_Song.pdf",
                            mime="application/pdf",
                            type="primary",
                            use_container_width=True
                        )
                    os.remove(pdf_file)
