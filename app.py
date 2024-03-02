import os
import streamlit as st
import tempfile
import fitz
from PyPDF2 import PdfFileWriter
from transformers.pipelines import pipeline

summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

def compress_pdf(input_path, target_size_mb):
    target_size_bytes = target_size_mb * 1024 * 1024

    input_size_bytes = os.path.getsize(input_path)

    if input_size_bytes <= target_size_bytes:
        return input_path

    output_path = os.path.join(os.path.dirname(input_path), "temp_output.pdf")

    pdf_document = fitz.open(input_path)

    pdf_writer = PdfFileWriter()

    compression_ratio = target_size_bytes / input_size_bytes

    for page_num in range(pdf_document.page_count):
        page = pdf_document[page_num]
        page_scale_factor = 1 / compression_ratio
        page.set_zoom(page_scale_factor)

        pdf_writer.addPage(page)

    pdf_document.close()

    with open(output_path, 'wb') as output_file:
        pdf_writer.write(output_file)

    return output_path

def extract_text_from_pdf(file):
    pdf_document = fitz.open(file)
    text = ""
    for page_num in range(pdf_document.page_count):
        page = pdf_document[page_num]
        text += page.get_text()

    pdf_document.close()
    return text

def process_in_chunks(text, chunk_size, max_length, min_length):
    # Tokenize text into smaller chunks
    chunks = [text[i:i + chunk_size] for i in range(0, len(text), chunk_size)]

    # Summarize each chunk individually
    summarized_chunks = []
    for chunk in chunks:
        adjusted_max_length = min(len(chunk) - 50, max_length)
        adjusted_max_length = max(adjusted_max_length, 100)  

        # print(f"Chunk length: {len(chunk)}")
        # print(f"Adjusted max_length: {adjusted_max_length}")

        summary = summarizer(chunk, max_length=adjusted_max_length, min_length=min_length)
        summarized_chunks.append(summary[0]['summary_text'])

    # Concatenate the summarized chunks without spaces
    full_summary = ''.join(summarized_chunks)
    return full_summary

def main():
    st.title("PDF Summarizer")

    uploaded_file = st.file_uploader("Upload a PDF file", type="pdf")

    if uploaded_file is not None:
        with st.spinner("Uploading and processing the file..."):
            # Create a temporary directory for storing files
            temp_dir = tempfile.TemporaryDirectory()
            temp_input_path = os.path.join(temp_dir.name, "temp.pdf")
            with open(temp_input_path, 'wb') as temp_input_file:
                temp_input_file.write(uploaded_file.read())

            target_size_mb = 5

            # Compress the PDF to the target size
            temp_output_path = compress_pdf(temp_input_path, target_size_mb)

            # Extract text from the compressed PDF
            text = extract_text_from_pdf(temp_output_path)

            # Define the chunk size
            chunk_size = 2000  

            # Process the text in chunks
            full_summary_chunks = process_in_chunks(text, chunk_size, max_length=200, min_length=30)

            # Display a loading message for summary generation
            with st.spinner("Generating the summary..."):
                if len(full_summary_chunks) > 1:
                    # If there are more than 1 chunks, concatenate and display
                    concatenated_summary = ''.join(full_summary_chunks)
                    st.subheader("Concatenated Summary:")
                    st.write(concatenated_summary)
                else:
                    # If there is only one chunk, display the summary of that chunk
                    st.subheader("Summary:")
                    st.write(full_summary_chunks[0])

            # remove temporary directory and its contents
            temp_dir.cleanup()

if __name__ == "__main__":
    main()
