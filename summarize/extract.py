import io
import logging
import os

import docx2txt
import pdfplumber
import pytesseract
from cdp_backend.pipeline import transcript_model
from pdf2image.exceptions import PDFInfoNotInstalledError
from pdf2image.pdf2image import convert_from_bytes

from .mime import split_content_type

logger = logging.getLogger(__name__)

# TODO the various _clean(...) methods are first-cut hacks built by looking
# at lots of PDF documents and coming up with some okay-ish heuristics for
# cleaning up their extracted text. I mostly looked at Seattle city council
# documents, so while these hacks work okay-ish elsewhere, it's really worth
# revisiting the whole question in the future.


def _truncate_str(s: str, length: int) -> str:
    return s[:length] + "..." if len(s) > length else s


def _clean_sequential_line_numbers_v1(text: str) -> str:
    """Try to find and remove sequential line numbers from a string. v1."""
    # In particular, we look for lines that start with a number and a space:
    #
    # 1 Some text that goes for a while.
    # 2 Some other text that goes for a while.
    # 3 Yet another text that goes for a while.
    #
    # We want to remove the numbers and the spaces, and remove newlines, so that
    # we get:
    #
    # Some text that goes for a while. Some other text that goes for a while.
    # Yet another text that goes for a while.
    #
    # Importantly, we only do this if the line starts with a number and a space,
    # and we see at least three lines that start with a number and a space. Numbers
    # must be sequential; if we don't see a matching number, we stop.
    #
    # This was motivated by Seattle ORDINANCE and RESOLUTION documents with
    # line numbers in their gutters, which line numbers pdfplumber dutifully
    # extracts.

    lines = text.splitlines()
    cleaned_lines = []

    # If true, we're in a numbered sequence.
    in_sequence = False

    # If in_sequence is true, the next number we expect.
    sequence_number = 0

    # If in_sequence is true, the text we've accumulated so far.
    sequence_line = ""

    # Walk through the lines of the text.
    for i, line in enumerate(lines):
        # If we're not in a sequence...
        if not in_sequence:
            # If this line starts with "1 " or is precisely "1", decide if we
            # should start a sequence by looking at the next two lines to see
            # if they start with "2 " and "3 ". (Be careful not to go out of
            # bounds.)
            if line.startswith("1 ") or line == "1":
                has_2 = i + 1 < len(lines) and (
                    lines[i + 1].startswith("2 ") or lines[i + 1] == "2"
                )
                has_3 = i + 2 < len(lines) and (
                    lines[i + 2].startswith("3 ") or lines[i + 2] == "3"
                )
                if has_2 and has_3:
                    in_sequence = True
                    sequence_number = 2  # we expect this next
                    sequence_line = line[len("1 ") :] if line != "1" else ""
            else:
                # This line doesn't start with "1 ", so just add it to the
                # cleaned lines.
                cleaned_lines.append(line)
        else:
            # We're in a sequence. If this line starts with the next number we
            # expect, add it to the sequence.
            if line.startswith(f"{sequence_number} ") or line == str(sequence_number):
                sequence_line += (
                    " " + line[len(f"{sequence_number} ") :]
                    if line != str(sequence_number)
                    else ""
                )
                sequence_number += 1
            else:
                # We're no longer in a sequence.
                cleaned_lines.append(sequence_line)
                in_sequence = False
                cleaned_lines.append(line)

    if in_sequence:
        cleaned_lines.append(sequence_line)

    return "\n".join(cleaned_lines)


def _clean_headers_footers_v1(text: str) -> str:
    """Clean common headers/footers found in extracted PDF content. v1."""
    # Look for a line that starts with "Template last revised"; if we find it,
    # remove that line and (assuming we don't go out of bounds), the next 3.
    # Again, this works for Seattle PDFs.
    lines = text.splitlines()
    strike_indexes = [
        i for i, line in enumerate(lines) if line.startswith("Template last revised")
    ]
    # Walk through the strike indexes in reverse order so that we don't
    # invalidate the indexes of the lines we're removing.
    for i in reversed(strike_indexes):
        if i + 3 < len(lines):
            del lines[i : i + 4]
    return "\n".join(lines)


def _clean_pdf(text: str) -> str:
    """Clean a string extracted from a PDF using pdfPlumber."""
    text = _clean_sequential_line_numbers_v1(text)
    text = _clean_headers_footers_v1(text)
    return text


def _extract_text(io: io.BytesIO, charset: str | None) -> str:
    """Extract text from a text document. Piece of cake!."""
    data = io.read()
    return data.decode(charset or "utf-8")


def _extract_pdf_machine_readable(io: io.BytesIO) -> str:
    """Extract text from a document using pdfPlumber. v1."""
    try:
        with pdfplumber.open(io) as pdf:
            texts = []
            for page in pdf.pages:
                text = page.extract_text()
                text = _clean_pdf(text)
                texts.append(text)
            return "\n".join(texts).strip()
    except Exception as e:
        short_e = _truncate_str(str(e), 64)
        return (
            f"Please ignore: unable to extract content from the PDF named '{short_e}'."
        )


TESSERACT_CMD: str | None = os.getenv("TESSERACT_CMD")
POPPLER_PATH: str | None = os.getenv("POPPLER_PATH")


def _extract_pdf_ocr(io: io.BytesIO) -> str:
    """Extract from a document using tesseract OCR. v1."""
    if not TESSERACT_CMD or not POPPLER_PATH:
        logger.info(
            "Would have used OCR on a PDF document, but "
            "TESSERACT_CMD or POPPLER_PATH not set."
        )
        return ""
    pytesseract.pytesseract.tesseract_cmd = TESSERACT_CMD
    try:
        text = ""
        pdf_pages = convert_from_bytes(io.read(), dpi=300, poppler_path=POPPLER_PATH)
        for page in pdf_pages:
            text += pytesseract.image_to_string(page)
    except PDFInfoNotInstalledError:
        logger.exception(
            "The `POPPLER_PATH` environment variable is not set correctly."
        )
        raise
    except pytesseract.pytesseract.TesseractNotFoundError:
        logger.exception(
            "The `TESSERACT_CMD` environment variable is not set correctly."
        )
        raise
    except Exception as e:
        short_e = _truncate_str(str(e), 64)
        return (
            f"Please ignore: unable to extract content from the PDF named '{short_e}'."
        )
    return text.strip()


def _extract_pdf(io: io.BytesIO) -> str:
    """Extract text from a PDF document. v1."""
    extracted = _extract_pdf_machine_readable(io)
    if not extracted:
        io.seek(0)
        extracted = _extract_pdf_ocr(io)
    return extracted


def _extract_msword(io: io.BytesIO) -> str:
    """Extract text from a Word document. v1."""
    try:
        return docx2txt.process(io)
    except Exception as e:
        short_e = _truncate_str(str(e), 64)
        return (
            f"Please ignore: unable to extract content from the DOCX named '{short_e}'."
        )


def extract_text_from_bytes(io: io.BytesIO, content_type: str) -> str:
    """Extract text from a document using a pipeline of extractors. v1."""
    mime_type, charset = split_content_type(content_type)
    if mime_type == "application/pdf":
        return _extract_pdf(io)
    elif mime_type == "text/plain":
        return _extract_text(io, charset)
    elif mime_type in {
        "application/msword",
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    }:
        return _extract_msword(io)
    else:
        raise ValueError(f"Currently unsupported MIME type {mime_type}.")


def extract_text_from_transcript_model(tm: transcript_model.Transcript) -> str:
    """Return the text of a CDP transcript model."""
    return "\n\n".join(sentence.text for sentence in tm.sentences)
