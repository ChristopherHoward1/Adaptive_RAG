import os
import PyPDF2
import docx
import markdown
from bs4 import BeautifulSoup
import nltk
import spacy
from fastcoref import spacy_component
import torch

#nltk.download('punk-tab')
nltk.download('punkt')
# Initialize SpaCy and add FastCoref to the pipeline
nlp = spacy.load("en_core_web_sm", exclude=["parser", "lemmatizer", "ner", "textcat"])
nlp.add_pipe(
    "fastcoref", 
    config={
        'model_architecture': 'LingMessCoref', 
        'model_path': 'biu-nlp/lingmess-coref', 
        'device': 'cuda' if torch.cuda.is_available() else 'cpu'
        })

def read_pdf(file_path):
    text = ''
    with open(file_path, 'rb') as file:
        reader = PyPDF2.PdfReader(file)
        for page_num in range(len(reader.pages)):
            page = reader.pages[page_num]
            page_text = page.extract_text()
            if page_text:
                text += page_text
    return text

def read_docx(file_path):
    doc = docx.Document(file_path)
    text = '\n'.join([para.text for para in doc.paragraphs])
    return text

def read_markdown(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        md_content = file.read()
    html = markdown.markdown(md_content)
    soup = BeautifulSoup(html, features="html.parser")
    text = soup.get_text()
    return text

def coref_resolution(text, nlp):
    """
    Use SpaCy with FastCoref to resolve coreferences in the provided text.
    
    Args:
        text (str): The input text to perform coreference resolution on.
        nlp (spacy.Language): SpaCy pipeline with FastCoref component.
    
    Returns:
        str: Text with resolved coreferences.
    """
    try:
        doc = nlp(text, component_cfg={"fastcoref": {'resolve_text': True}})

        if not doc._.coref_clusters:
            print("No coreference clusters found. Returning original text.")
            return text

        
        resolved_text = doc._.resolved_text
        return resolved_text
    
    except IndexError as e:
        print(f"IndexError occurred: {e}. Returning original text.")
        return text

    except Exception as e:
        print(f"An unexpected error occurred during coreference resolution: {e}")
        return text



def process_file(file_path):
    _, file_extension = os.path.splitext(file_path)
    file_extension = file_extension.lower()

    if file_extension == '.pdf':
        raw_text = read_pdf(file_path)
    elif file_extension == '.docx':
        raw_text = read_docx(file_path)
    elif file_extension in ['.md', '.markdown']:
        raw_text = read_markdown(file_path)
    else:
        raise ValueError(f"Unsupported file extension: {file_extension}")

    resolved_text = coref_resolution(raw_text, nlp)
    return resolved_text