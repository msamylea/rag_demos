from pinecone import Pinecone, ServerlessSpec, Index
import os
from dotenv import load_dotenv
from pathlib import Path
import docx2txt
from sentence_transformers import SentenceTransformer
import zipfile
from glob import glob
import re
import os
import win32com.client as win32
from win32com.client import constants
import unicodedata


# Load environment variables
env_path = Path('.') / '.env'
load_dotenv(dotenv_path=env_path)

# Initialize Pinecone
pc = Pinecone(api_key=os.environ.get('PINECONE_API_KEY'))

index_name = "mepolicy"
index = pc.Index(index_name)


def sanitize_filename(filename):
    # Normalize the filename to remove non-ASCII characters
    return unicodedata.normalize('NFKD', filename).encode('ascii', 'ignore').decode('ascii')

def ensure_makepy():
    try:
        win32.gencache.EnsureDispatch('Word.Application')
    except Exception:
        import subprocess
        subprocess.run(['python', '-m', 'win32com.client.makepy', 'Microsoft Word 16.0 Object Library'])

def read_and_parse_files():
    embed_model = SentenceTransformer("all-mpnet-base-v2")
    file_paths = [
        "benefits_manual",
        "emergency_rules",
        "recently_adopted_rules",
    ]

    ensure_makepy()

    for file_path in file_paths:
        for file in os.listdir(file_path):
            sanitized_file = sanitize_filename(file)
            full_path = f"{file_path}/{file}"
            
            if file.endswith(".txt"):
                with open(full_path, "r", encoding='utf-8') as f:
                    text = f.read()
            elif file.endswith(".docx"):
                try:
                    text = docx2txt.process(full_path)
                except zipfile.BadZipFile:
                    print(f"Failed to process {file}: Not a valid docx file")
                    continue
            elif file.endswith(".doc"):
                try:
                    import shutil
                    gen_path = win32.gencache.GetGeneratePath()
                    shutil.rmtree(gen_path, ignore_errors=True)
                    win32.gencache.is_readonly = False
                    win32.gencache.Rebuild()

                    word = win32.gencache.EnsureDispatch('Word.Application')
                    doc = word.Documents.Open(os.path.abspath(f"{file_path}/{file}"))
                    doc.Activate()

                    # Rename path with .docx
                    new_file_abs = re.sub(r'\.\w+$', '.docx', os.path.abspath(f"{file_path}/{file}"))

                    # Save and Close
                    word.ActiveDocument.SaveAs(
                        new_file_abs, FileFormat=constants.wdFormatXMLDocument
                    )
                    doc.Close(False)
                    word.Quit()

                    text = docx2txt.process(new_file_abs)
                except Exception as e:
                    print(f"Failed to process {file}: {e}")
                    continue
            else:
                print(f"Failed to process {file}: Not a valid file type")
                continue
            
            # Create embeddings
            embeddings = embed_model.encode([text])
            
            # Prepare metadata
            metadata = {
                "filename": file,
                "file_path": full_path,
                "content": text[:1000]  # Store first 1000 characters of content
            }
            
            # Upsert embeddings into Pinecone index with metadata
            index.upsert(vectors=[(sanitized_file, embeddings[0].tolist(), metadata)])
            print(f"Processed and upserted file: {file}")

# Call the function to read and parse files
read_and_parse_files()