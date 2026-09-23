from pathlib import Path

import pymupdf
from llama_index.core import Document


PDF_PATH = Path("data/pdfs")


def load_pdfs():
    documents = []

    for pdf_file in PDF_PATH.glob("*.pdf"):
        pdf = pymupdf.open(pdf_file)

        text = ""

        for page in pdf:
            text += page.get_text()

        pdf.close()

        if text.strip():
            document = Document(
                text=text,
                metadata={
                    "file_name": pdf_file.name,
                    "file_path": str(pdf_file),
                },
            )

            documents.append(document)

    return documents