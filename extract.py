import fitz  # PyMuPDF for PDF handling
import re
import json

def extract_answers_from_pdf(pdf_path, questions):
    """
    Extracts answers to questions from a PDF file.

    Args:
        pdf_path (str): Path to the PDF file.
        questions (list): List of questions to find answers for.

    Returns:
        dict: A dictionary with questions as keys and extracted answers as values.
    """
    # Read the PDF file and extract text
    with fitz.open(pdf_path) as doc:
        text = ""
        for page in doc:
            text += page.get_text()

    # Combine questions into a regex pattern to use them as delimiters
    question_pattern = "|".join([re.escape(q) for q in questions])

    # Split text into sections using the questions as delimiters
    sections = re.split(question_pattern, text, flags=re.IGNORECASE)

    # Pair each question with the corresponding section of text
    answers = {}
    for idx, question in enumerate(questions):
        if idx + 1 < len(sections):
            # The answer is the text between the current question and the next
            answers[question] = sections[idx + 1].strip()
        else:
            # If no more sections, assign empty string
            answers[question] = ""

    return answers

# Path to the grant application PDF file
pdf_path = "/mnt/data/2025 CLG Application Final Clean Version (1).pdf"  # Update with your file path

# Predefined list of questions (or load them from the JSON file created earlier)
questions = [
    "1. Primary Contact First Name",
    "2. Primary Contact Last Name",
    "3. Primary Contact Telephone",
    "4. Landline or cell?",
    "5. Primary Contact Email",
    "6. Primary Contact home address",
    "7. Please indicate the race or ethnicity of the primary contact (choose all that apply)"
]

# Extract answers from the PDF
answers = extract_answers_from_pdf(pdf_path, questions)

# Save answers to a JSON file
answers_output_file = "grant_application_answers.json"
with open(answers_output_file, "w", encoding="utf-8") as f:
    json.dump(answers, f, indent=4)

print(f"Extracted answers saved to {answers_output_file}")
