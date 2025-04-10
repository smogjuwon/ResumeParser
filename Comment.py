# Importing required libraries
import spacy  # For NLP and Named Entity Recognition
import pandas as pd  # For data manipulation and creating DataFrames
import pytesseract  # OCR tool to extract text from images
import pypdfium2 as pdfium  # To handle PDF rendering
from PIL import Image  # For image processing
import re  # Regular expressions for pattern matching
import streamlit as st  # To build interactive web apps
from io import BytesIO  # Handle byte streams for in-memory file processing
import gdown  # To download files from Google Drive
import os  # To check file paths and directories

# Define model URL and path
MODEL_URL = "https://drive.google.com/drive/folders/1usHLXJTdU1eLIAKvjzLtCtauYJEC8Qqw?usp=sharing"
MODEL_PATH = "my_spacy_model"

# Download the model if not already present
if not os.path.exists(MODEL_PATH):
    gdown.download(MODEL_URL, MODEL_PATH, quiet=False)

# Load the spaCy model
nlp = spacy.load(MODEL_PATH)

# Cache the loaded model using Streamlit for performance
@st.cache_resource
def load_model():
    return spacy.load(MODEL_PATH)

nlp = load_model()

# Function to convert PDF pages into images
def process_pdf(resume, scale=300/72):
    file = pdfium.PdfDocument(resume)  # Open the PDF file
    page_indices = [i for i in range(len(file))]  # Get all page indices

    # Render pages to PIL images
    renderer = file.render(
        pdfium.PdfBitmap.to_pil,
        page_indices=page_indices,
        scale=scale,
    )

    list_final_images = []

    # Convert each page image to byte format
    for i, image in zip(page_indices, renderer):
        image_byte_array = BytesIO()
        image.save(image_byte_array, format='jpeg', optimize=True)
        image_byte_array = image_byte_array.getvalue()
        list_final_images.append({i: image_byte_array})

    return list_final_images

# Set path for Tesseract OCR (update as needed)
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# Extract text from image bytes using Tesseract OCR
from pytesseract import image_to_string
def extract_text_from_image(extract1):
    image_list = [list(data.values())[0] for data in extract1]  # Get image byte arrays
    image_content = []

    for index, image_bytes in enumerate(image_list):
        extract1 = Image.open(BytesIO(image_bytes))  # Convert bytes to image
        raw_text = str(image_to_string(extract1))  # Extract text
        image_content.append(raw_text)

    return "\n".join(image_content)  # Combine all page texts

# Use regex to extract key info like email, phone, LinkedIn, and skills
def extract_info_with_regex(text):
    extracted_info = {}

    # Extract email address
    emails = re.findall(r'[\w\.-]+@[\w\.-]+', text)
    extracted_info['Email'] = emails[0] if emails else None

    # Extract phone numbers in various formats
    phones = re.findall(r'\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}', text)
    extracted_info['Phone'] = phones[0] if phones else None

    # Extract LinkedIn profile URLs
    linkedin = re.findall(r'https?://(?:www\.)?linkedin\.com/in/[\w-]+', text)
    extracted_info['LinkedIn'] = linkedin[0] if linkedin else None

    # Get the first line (commonly a name, but not stored here)
    first_line = text.strip().split("\n")[0]

    # Define a list of skills to check against
    skills_list = [
        "Python", "Machine Learning", "NLP", "TensorFlow", 
        "Communication", "SQL", "PowerBI", "Tableau", "Teamwork"
    ]

    # Check if skills are present in text
    extracted_skills = [skill for skill in skills_list if skill in text]

    return extracted_info  # Skills are detected but not returned here — optional to include

# Apply spaCy NER and merge it with regex extracted info
def perform_ner(text):
    doc = nlp(text)  # Process text with spaCy
    entity_dict = {ent.label_: ent.text for ent in doc.ents}  # Extract entities

    entity_dict.update(extract_info_with_regex(text))  # Merge with regex results

    return entity_dict

# Initialize a DataFrame in session state if not already there
if "resume_df" not in st.session_state:
    st.session_state.resume_df = pd.DataFrame()

# Streamlit web interface
st.title("Resume Parser with Named Entity Recognition")
st.write("Upload a PDF resume to extract and store named entities.")

# Upload a resume PDF
pdf_file = st.file_uploader("Choose a PDF file", type=["pdf"])

# If a file is uploaded, process and extract information
if pdf_file is not None:
    with st.spinner("Processing the PDF..."):
        image = process_pdf(pdf_file)  # Convert PDF to image
        extracted_text = extract_text_from_image(image)  # Extract text using OCR

    st.subheader("Extracted Text")
    st.text_area("Text from PDF", extracted_text, height=300)  # Show extracted text

    # Perform NER and store results
    entity_dict = perform_ner(extracted_text)

    if entity_dict:
        df_new = pd.DataFrame([entity_dict])
        st.session_state.resume_df = pd.concat(
            [st.session_state.resume_df, df_new], ignore_index=True
        )
        st.success("Entities extracted and saved!")

    # Show all parsed resumes
    st.subheader("Stored Resumes")
    st.dataframe(st.session_state.resume_df)

    # Allow downloading results as CSV
    st.download_button(
        label="Download CSV",
        data=st.session_state.resume_df.to_csv(index=False).encode("utf-8"),
        file_name="parsed_resumes.csv",
        mime="text/csv",
    )
