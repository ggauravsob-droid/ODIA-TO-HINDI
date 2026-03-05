import streamlit as st
import pytesseract
import cv2
import numpy as np
import pdfkit
import os
from indic_transliteration import sanscript
from indic_transliteration.sanscript import transliterate

# --- APP CONFIG & SESSION STATE ---
st.set_page_config(page_title="Odia Song OCR & Translator", layout="wide")

# Session state to hold our text so it doesn't disappear when typing
if 'extracted_text' not in st.session_state:
    st.session_state.extracted_text = ""

st.title("🎵 Odia Song OCR & Multilingual PDF Generator")
st.write("Upload an image, extract the text, fix any minor matra mistakes, and generate a formatted PDF in Odia, Hindi, and English.")

# --- CORE FUNCTIONS ---
def clean_image(image_bytes):
    """Advanced image processing to preserve tiny matras and complex Odia conjuncts."""
    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    # 1. Upscale the image by 2.5x to make tiny details larger for the OCR
    img_enlarged = cv2.resize(img, None, fx=2.5, fy=2.5, interpolation=cv2.INTER_CUBIC)
    
    # 2. Convert to Grayscale
    gray = cv2.cvtColor(img_enlarged, cv2.COLOR_BGR2GRAY)
    
    # 3. Apply CLAHE (Smart contrast boost) to keep thin strokes visible
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    enhanced_gray = clahe.apply(gray)
    
    temp_path = "temp_image.png"
    cv2.imwrite(temp_path, enhanced_gray)
    return temp_path

def perform_ocr(image_path):
    """Extracts Odia text using the Neural Network LSTM engine."""
    # --oem 1 forces Deep Learning LSTM model. preserve_interword_spaces keeps the layout.
    custom_config = r'-c preserve_interword_spaces=1 --oem 1 --psm 6'
    text = pytesseract.image_to_string(image_path, lang='ori', config=custom_config)
    return text

def transliterate_text(odia_text):
    hindi_script = transliterate(odia_text, sanscript.ORIYA, sanscript.DEVANAGARI)
    eng_script = transliterate(odia_text, sanscript.ORIYA, sanscript.IAST)
    return hindi_script, eng_script

def create_pdf(odia_text, hindi_text, eng_text, output_filename="song.pdf"):
    """Generates the layout-preserved PDF."""
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="UTF-8">
        <link href="https://fonts.googleapis.com/css2?family=Noto+Sans+Devanagari&family=Noto+Sans+Oriya&family=Noto+Sans&display=swap" rel="stylesheet">
        <style>
            body {{ padding: 20px; font-family: 'Noto Sans', sans-serif; }}
            h2 {{ color: #333; border-bottom: 2px solid #ddd; padding-bottom: 5px; }}
            .text-block {{ white-space: pre-wrap; font-size: 16px; line-height: 1.5; margin-bottom: 40px; }}
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
    
    options = {
        'encoding': "UTF-8",
        'enable-local-file-access': None,
        'quiet': ''
    }
    
    pdfkit.from_string(html_content, output_filename, options=options)
    return output_filename

# --- UI WORKFLOW ---
uploaded_file = st.file_uploader("1. Upload Image (JPG/PNG)", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.image(uploaded_file, caption="Uploaded Image", use_container_width=True)
        if st.button("Scan & Extract Text", type="primary"):
            with st.spinner("Enhancing image and extracting text..."):
                temp_image_path = clean_image(uploaded_file.getvalue())
                st.session_state.extracted_text = perform_ocr(temp_image_path)
                os.remove(temp_image_path) # Cleanup
                st.success("Scan Complete! Please review the text on the right.")

    with col2:
        if st.session_state.extracted_text:
            st.info("💡 **Tip:** If the OCR missed a complex matra (like 'pra' or 'nda'), you can manually correct it in the box below before generating the PDF.")
            
            # User can edit the OCR output here!
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
                    os.remove(pdf_file) # Cleanup
