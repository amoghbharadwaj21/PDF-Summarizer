import os
import streamlit as st
import fitz
from PyPDF2 import PdfFileWriter
from transformers.pipelines import pipeline

# Load the summarization pipeline with the BART model
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

def compress_pdf(input_path, target_size_mb):
    # Calculate the target size in bytes
    target_size_bytes = target_size_mb * 1024 * 1024

    # Get the current size of the PDF
    input_size_bytes = os.path.getsize(input_path)

    if input_size_bytes <= target_size_bytes:
        # No need to compress if the file is already smaller than the target size
        return input_path

    # Determine the output path for the compressed PDF in the same directory
    output_path = os.path.join(os.path.dirname(input_path), "temp_output.pdf")

    # Use PyMuPDF to open the input PDF
    pdf_document = fitz.open(input_path)

    # Create a PdfFileWriter object for writing the compressed PDF
    pdf_writer = PdfFileWriter()

    # Calculate the compression ratio
    compression_ratio = target_size_bytes / input_size_bytes

    # Iterate through each page of the PDF
    for page_num in range(pdf_document.page_count):
        page = pdf_document[page_num]
        page_scale_factor = 1 / compression_ratio
        page.set_zoom(page_scale_factor)

        # Add the page to the PdfFileWriter
        pdf_writer.addPage(page)

    # Close the PyMuPDF document
    pdf_document.close()

    # Write the compressed PDF to the output file
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
        # Adjust max_length based on the length of the chunk
        adjusted_max_length = min(len(chunk) - 50, max_length)
        adjusted_max_length = max(adjusted_max_length, 100)  # Set a minimum value for adjusted_max_length

        # Print statements for debugging
        print(f"Chunk length: {len(chunk)}")
        print(f"Adjusted max_length: {adjusted_max_length}")

        summary = summarizer(chunk, max_length=adjusted_max_length, min_length=min_length)
        summarized_chunks.append(summary[0]['summary_text'])

    # Concatenate the summarized chunks without spaces
    full_summary = ''.join(summarized_chunks)
    return full_summary

def main():
    st.title("PDF Summarizer with Streamlit")

    uploaded_file = st.file_uploader("Upload a PDF file", type="pdf")

    if uploaded_file is not None:
        # Display a loading message for file uploading
        with st.spinner("Uploading and processing the file..."):
            # Set the target size limit to 5MB
            target_size_mb = 5

            # Create a file for the input PDF
            temp_input_path = os.path.join(os.path.dirname(__file__), "temp.pdf")
            with open(temp_input_path, 'wb') as temp_input_file:
                temp_input_file.write(uploaded_file.read())

            # Compress the PDF to the target size
            temp_output_path = compress_pdf(temp_input_path, target_size_mb)

            # Extract text from the compressed PDF
            text = extract_text_from_pdf(temp_output_path)

            # Define the chunk size
            chunk_size = 2000  # Adjust the chunk size according to your preference

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

            # Clean up: remove temporary files
            os.remove(temp_input_path)
            # os.remove(temp_output_path)

if __name__ == "__main__":
    main()
