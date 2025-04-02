import spacy
import pandas as pd
import pytesseract
import pypdfium2 as pdfium
from PIL import Image
import re
import streamlit as st
from io import BytesIO

import gdown
import os
import spacy

MODEL_URL = "https://drive.google.com/drive/folders/1VWN9tEM903fu42ssG5wsrNdSmFJMTTTW?usp=sharing"
MODEL_PATH = "my_spacy_model"

if not os.path.exists(MODEL_PATH):
    gdown.download(MODEL_URL, MODEL_PATH, quiet=False)

nlp = spacy.load(MODEL_PATH)


@st.cache_resource
# Load your custom model
#nlp = spacy.load('my_spacy_model')
nlp = spacy.load(MODEL_PATH)

# Function to process the PDF and extract text
def process_pdf(resume, scale=300/72):

    file = pdfium.PdfDocument(resume)
    page_indices = [i for i in range(len(file))]

    renderer = file.render(
        pdfium.PdfBitmap.to_pil,
        page_indices = page_indices,
        scale = scale,
    )

    list_final_images = []

    for i, image in zip(page_indices, renderer):

        image_byte_array = BytesIO()
        image.save(image_byte_array, format='jpeg', optimize=True)
        image_byte_array = image_byte_array.getvalue()
        list_final_images.append(dict({i:image_byte_array}))

    return list_final_images



# Specify the path to the Tesseract executable (replace with your actual path)
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'


from pytesseract import image_to_string
# Function to extract text from images using Tesseract
def extract_text_from_image(extract1):
    image_list = [list(data.values())[0] for data in extract1]
    image_content = []

    for index, image_bytes in enumerate(image_list):
        extract1 = Image.open(BytesIO(image_bytes))
        raw_text = str(image_to_string(extract1))
        image_content.append(raw_text)

    return "\n".join(image_content)


# Function to extract additional information using regex
def extract_info_with_regex(text):
    extracted_info = {}

    # Extract email addresses
    emails = re.findall(r'[\w\.-]+@[\w\.-]+', text)
    extracted_info['Email'] = emails[0] if emails else None

    # Extract phone numbers (supports various formats)
    phones = re.findall(r'\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}', text)
    extracted_info['Phone'] = phones[0] if phones else None

    # Extract LinkedIn profiles
    linkedin = re.findall(r'https?://(?:www\.)?linkedin\.com/in/[\w-]+', text)
    extracted_info['LinkedIn'] = linkedin[0] if linkedin else None

    # Extract first line (assuming it's the name)
    first_line = text.strip().split("\n")[0]

    # Predefined list of skills
    skills_list = ["Python", "Machine Learning", "NLP", "TensorFlow", "Communication", "SQL", "PowerBI", "Tableau",
                   "Teamwork"]

    # Extract skills by checking for matches in the skills list
    extracted_skills = [skill for skill in skills_list if skill in text]

    return extracted_info





# Function to perform NER with spaCy
def perform_ner(text):
    doc = nlp(text)
    entity_dict = {ent.label_: ent.text for ent in doc.ents}

    # Merge regex-extracted information
    entity_dict.update(extract_info_with_regex(text))

    return entity_dict


# Initialize or load stored DataFrame
if "resume_df" not in st.session_state:
    st.session_state.resume_df = pd.DataFrame()

# Streamlit UI
st.title("Resume Parser with Named Entity Recognition")
st.write("Upload a PDF resume to extract and store named entities.")

# File uploader
pdf_file = st.file_uploader("Choose a PDF file", type=["pdf"])

if pdf_file is not None:
    with st.spinner("Processing the PDF..."):
        image = process_pdf(pdf_file)
        extracted_text = extract_text_from_image(image)

    st.subheader("Extracted Text")
    st.text_area("Text from PDF", extracted_text, height=300)

    # Perform NER and store results
    entity_dict = perform_ner(extracted_text)

    if entity_dict:
        df_new = pd.DataFrame([entity_dict])
        st.session_state.resume_df = pd.concat([st.session_state.resume_df, df_new], ignore_index=True)
        st.success("Entities extracted and saved!")

    # Display stored resumes
    st.subheader("Stored Resumes")
    st.dataframe(st.session_state.resume_df)

    # Option to download CSV
    st.download_button(
        label="Download CSV",
        data=st.session_state.resume_df.to_csv(index=False).encode("utf-8"),
        file_name="parsed_resumes.csv",
        mime="text/csv",
    )
