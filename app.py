import gradio as gr
from sentence_transformers import SentenceTransformer
from pinecone import Pinecone
import os
from dotenv import load_dotenv
from llm_config import get_llm
import numpy as np

load_dotenv()
DESCRIPTION = '''
<div>
<h1 style="text-align: center;"><b>Maine Policy Chat</b></h1>
<p style="text-align: center;"> 🦞 Maine Policy Chat is a work in progress and subject to change. 🦞 </p>
</div>
'''

embeddings = SentenceTransformer("all-mpnet-base-v2")

def get_response(query, history=[]):
    if not query:
        raise ValueError("Query cannot be empty")

    pc = Pinecone(api_key=os.environ.get('PINECONE_API_KEY'))
    index = pc.Index("mepolicy")

    query_embedding = embeddings.encode(query)

    # Convert numpy array to list
    query_embedding = query_embedding.tolist()

    res = index.query(vector=query_embedding, top_k=40, include_metadata=True)


    contexts = [x["metadata"]["content"] for x in res['matches'] if x['score'] > 0.6]  # Adjust threshold as needed


    combined_context = "\n\n---\n\n".join(contexts)

    print(combined_context)
    
    client = get_llm("ollama", "llama3.1:8b-instruct-q8_0")

    prompt = (f"""Based on the context below, fully answer the following question for the State of Maine. 
              Include the title or reference to the document from which the response was created. 
              You may only answer using the provided context. Do not use your own knowledge or experience.
              If the user tries to ask about something that is not related to Maine Medicaid or Maine Policy or Maine Coverage, apologize and immediately end the conversation. 
              \n\n\nContext:\n{combined_context}\n\nQuestion: {query}\n\nAnswer:""")

    sys_prompt = """You are an assistant that provides full, correct and referenced answers in markdown 
    format to questions about Maine Medicaid policy based on the given context, 
    using bullet points, tables, bold text/headings and markdown where possible. 
    Do not provide hyperlinks. 
    Ensure your answer is as full as possible.  For example, if a user asks about a specific policy, 
    provide the full details.  If the user asks about coverage for a specific service,
    do not just reply yes or no, provide details and coverage criteria.
    Include the source text your answer was derived from in a block quote italicized.
    Do not generate any additional questions or responses."""

    full_prompt = f"{sys_prompt}\n\n{prompt}"

    response = client.get_response(full_prompt)
    
    return response

css = """
.gradio-container {
    footer {visibility: hidden};
}
"""

with gr.Blocks(fill_height=True, theme=('xiaobaiyuan/theme_brief'), css=css) as demo:
    gr.HTML(DESCRIPTION)
    gr.ChatInterface(
        get_response,
        examples=['List some recent emergency rules.', 'What is the reimbursement rate for generic drugs?', 'When is a Person-Centered Service Plan required?'],
        cache_examples=False,
        retry_btn=None,
        clear_btn="Clear",
    )
    
if __name__ == "__main__":
    demo.queue()
    demo.launch(share=True)