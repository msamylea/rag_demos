import streamlit as st
import pinecone
import asyncio
import psycopg2
from psycopg2 import Binary
from io import BytesIO
import fitz  # PyMuPDF
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from dotenv import load_dotenv
from pathlib import Path
from pinecone import Pinecone, ServerlessSpec
from llm_config import get_llm
from sentence_transformers import SentenceTransformer
import numpy as np
import base64

st.set_page_config(page_title="ME RAG System with Custom LLM", layout="wide")

env_path = Path('.') / '.env'
load_dotenv(dotenv_path=env_path)

# Initialize LLM
llm = get_llm("ollama", "llama3.1:8b-instruct-q8_0")

# Initialize Pinecone
pc = Pinecone(api_key=os.environ.get('PINECONE_API_KEY'))
index_name = "maine-policy"
index = pc.Index(index_name)

# Initialize SentenceTransformer
embeddings_model = SentenceTransformer("all-mpnet-base-v2")

with open( "style.css" ) as css:
    st.markdown( f'<style>{css.read()}</style>' , unsafe_allow_html= True)

st.markdown(""" <style>
            
#MainMenu {{visibility: hidden;}}
footer {{visibility: hidden;}}

    }
          
</style> """, unsafe_allow_html=True)


banner_html = """
    <header>
    <div class="wrapper">
        <h1>Maine Medicaid Knowledgebase<br> <span></h1>
    
    </header>
    """

st.html(banner_html)


def create_index():
    # Create Pinecone index if it doesn't exist
    if index_name not in pc.list_indexes().names():
        index = pc.create_index(
            name=index_name,
            dimension=768,
            metric="cosine",
            spec=ServerlessSpec(
                cloud='aws',
                region='us-east-1'
            )
        )
    else:
        index = pc.Index(index_name)
    return index

# File paths for PDFs
file_paths = [
    "emergency_rules",
    "recently_adopted_rules",
    "benefits_manual",
]

# PostgreSQL connection
conn = psycopg2.connect("dbname=full_docs user=postgres password=deeto host=localhost")
cur = conn.cursor()

def create_db():
    # Create tables if not exists
    cur.execute("""
        CREATE TABLE IF NOT EXISTS documents (
            id SERIAL PRIMARY KEY,
            name TEXT,
            content BYTEA
        )
    """)
    cur.execute("""
        CREATE TABLE IF NOT EXISTS document_chunks (
            id SERIAL PRIMARY KEY,
            document_id INTEGER REFERENCES documents(id),
            chunk_index INTEGER,
            chunk_text TEXT,
            vector_id TEXT
        )
    """)
    conn.commit()

# Function to process a single PDF
def process_pdf(file_path):
    with open(file_path, "rb") as file:
        pdf_bytes = file.read()
    
    # Store in PostgreSQL
    cur.execute("INSERT INTO documents (name, content) VALUES (%s, %s) RETURNING id",
                (os.path.basename(file_path), Binary(pdf_bytes)))
    doc_id = cur.fetchone()[0]
    conn.commit()
    
    # Process PDF content
    pdf_document = fitz.open(stream=pdf_bytes, filetype="pdf")
    text_content = ""
    for page in pdf_document:
        text_content += page.get_text()
    
    # Split text into chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = text_splitter.split_text(text_content)
    
    # Generate embeddings
    vector_embeddings = embeddings_model.encode(chunks)
    
    # Store in Pinecone and link to PostgreSQL
    for i, (chunk, embedding) in enumerate(zip(chunks, vector_embeddings)):
        vector_id = f"doc_{doc_id}_chunk_{i}"
        # Convert numpy array to list
        embedding_list = embedding.tolist()
        index.upsert(vectors=[(vector_id, embedding_list, {"text": chunk, "document_id": doc_id})])
        
        # Store chunk info in PostgreSQL
        cur.execute("""
            INSERT INTO document_chunks (document_id, chunk_index, chunk_text, vector_id)
            VALUES (%s, %s, %s, %s)
        """, (doc_id, i, chunk, vector_id))
    
    conn.commit()
    return doc_id

# Streamlit interface

# Process PDFs
# if st.button("Process PDFs"):
#     for path in file_paths:
#         for file in os.listdir(path):
#             if file.endswith(".pdf"):
#                 file_path = os.path.join(path, file)
#                 doc_id = process_pdf(file_path)
#                 st.success(f"Processed: {file}")

results = None

with st.container(border=True):
    col1, col2 = st.columns([2,2])
    with col1:
        with st.container(border=True):
            innercol1, innercol2 = st.columns([3, 1])
            with innercol1:
                query = st.text_input("Enter your query:")
            with innercol2:
                query_btn = st.button("Submit Query")
            if query_btn:
                query_embedding = embeddings_model.encode(query)
                query_embedding_list = query_embedding.tolist()
                results = index.query(vector=query_embedding_list, top_k=5, include_metadata=True)
                print("RAW RESULTS:", results)
                if results and results.get('matches'):
                    context = "\n\n".join([f"Document: {result['metadata'].get('document_name', 'Unknown')}\n{result['metadata']['text']}" for result in results['matches']])
                    
                    prompt = f"""Based on the following context from Maine Medicaid policy documents, provide a comprehensive and accurate answer to the question. 
                    Use specific details, quotations, and references from the provided context.
                    If the information is not available in the context, state that clearly.
                    Do not use any external knowledge or make assumptions.
                    
                    Context:
                    {context}
                    
                    Question: {query}
                    
                    Answer:"""

                    sys_prompt = """You are an expert assistant on Maine Medicaid policy. Your role is to provide detailed, accurate, and well-structured answers based solely on the given context. Follow these guidelines:

                    1. Use markdown formatting for clarity and readability.
                    2. Include relevant quotations from the source documents, using blockquotes.
                    3. Clearly cite the document names or sections when referencing information.
                    4. If the query cannot be fully answered with the given context, clearly state what information is missing.
                    5. Provide a comprehensive answer that covers all aspects of the query, if possible.
                    6. Use bullet points or numbered lists for multiple points or steps.
                    7. If appropriate, summarize key points at the end of your response.
                    8. Do not provide information or opinions beyond what is given in the context.
                    9. If the query is not related to Maine Medicaid policy, politely state that you can only answer questions on that topic."""

                    full_prompt = f"{sys_prompt}\n\n{prompt}"
                    
                    async def process_response():
                        async for chunk in llm.get_aresponse(full_prompt):
                            yield chunk

                    with st.spinner("Generating answer..."):
                        answer_container = st.empty()
                        
                        async def stream_response():
                            answer = ""
                            buffer = ""
                            async for chunk in process_response():
                                buffer += chunk
                                if len(buffer.split()) >= 10 or len(buffer) >= 100:
                                    answer += buffer
                                    answer_container.markdown(answer)
                                    buffer = ""
                            if buffer:
                                answer += buffer
                                answer_container.markdown(answer)
                            return answer

                        full_answer = asyncio.run(stream_response())
                    
                else:
                    st.write("No results found for your query. Please try a different question.")
                    
    with col2:
        with st.container(border=True):
            st.subheader("Source Document Result")
            if results and results.get('matches'):
                # Get the most relevant document (first match)
                most_relevant_match = results['matches'][0]
                doc_id = most_relevant_match['metadata']['document_id']
                chunk_text = most_relevant_match['metadata']['text']
                
                cur.execute("SELECT name FROM documents WHERE id = %s", (doc_id,))
                doc_name = cur.fetchone()[0]
                st.write(f"Most relevant document: {doc_name}")

                cur.execute("SELECT content FROM documents WHERE id = %s", (doc_id,))
                pdf_bytes = cur.fetchone()[0]
                    
                if pdf_bytes:
                    # Convert to bytes if it's a memoryview
                    if isinstance(pdf_bytes, memoryview):
                        pdf_bytes = pdf_bytes.tobytes()
                    
                    # Create PDF with highlights
                    pdf_document = fitz.open(stream=pdf_bytes, filetype="pdf")
                    for page in pdf_document:
                        areas = page.search_for(chunk_text)
                        for area in areas:
                            highlight = page.add_highlight_annot(area)
                            # Break after finding the first match
                            break
                        if areas:
                            # Break after highlighting in the first page with a match
                            break

                    # Save highlighted PDF
                    output = BytesIO()
                    pdf_document.save(output)
                    pdf_document.close()

                    # Encode the PDF content
                    base64_pdf = base64.b64encode(output.getvalue()).decode('utf-8')

                    # Display the PDF
                    pdf_display = f'<iframe src="data:application/pdf;base64,{base64_pdf}" width="100%" height="800px" type="application/pdf"></iframe>'
                    st.markdown(pdf_display, unsafe_allow_html=True)


                else:
                    st.write(f"PDF content for {doc_name} not found.")
            else:
                st.write("No source documents to display. Please enter a query first.")
    # Close database connection
cur.close()
conn.close()