
import gradio as gr
from langchain_core.runnables import RunnablePassthrough, RunnableParallel
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_openai import ChatOpenAI
import os

# --- Configuration --- #
# In a real deployment, consider using environment variables for sensitive keys
# For Hugging Face Spaces, you can add your OPENAI_API_KEY as a Space Secret.
# os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY") # For deployment, get from environment variable

if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY environment variable not set. Please set it in your Hugging Face Space secrets.")

# --- Initialize LLM --- #
llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.7, openai_api_key=OPENAI_API_KEY)

# --- Initialize Embeddings Model --- #
# This needs to be the same model used during vector store creation
embeddings_model_for_retrieval = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")

# --- Load Persisted ChromaDB Vector Stores --- #
# Ensure these directories are present in your Hugging Face Space

# Check if the directories exist
if not os.path.exists('./chroma_db_scripts'):
    print("Warning: chroma_db_scripts directory not found. Retriever for scripts will be empty.")
    vectorstore_scripts_loaded = None
else:
    vectorstore_scripts_loaded = Chroma(persist_directory='./chroma_db_scripts', embedding_function=embeddings_model_for_retrieval)

if not os.path.exists('./chroma_db_pdf'):
    print("Warning: chroma_db_pdf directory not found. Retriever for PDF will be empty.")
    vectorstore_pdf_loaded = None
else:
    vectorstore_pdf_loaded = Chroma(persist_directory='./chroma_db_pdf', embedding_function=embeddings_model_for_retrieval)

if not os.path.exists('./chroma_db_web_scraped'):
    print("Warning: chroma_db_web_scraped directory not found. Retriever for web-scraped data will be empty.")
    vectorstore_web_scraped_loaded = None
else:
    vectorstore_web_scraped_loaded = Chroma(persist_directory='./chroma_db_web_scraped', embedding_function=embeddings_model_for_retrieval)

# --- Create Retrievers --- #
retriever_scripts = vectorstore_scripts_loaded.as_retriever() if vectorstore_scripts_loaded else (lambda x: [])
retriever_pdf = vectorstore_pdf_loaded.as_retriever() if vectorstore_pdf_loaded else (lambda x: [])
retriever_web_scraped = vectorstore_web_scraped_loaded.as_retriever() if vectorstore_web_scraped_loaded else (lambda x: [])

# --- Combine Retrievers --- #
combined_retriever_runnable = (
    RunnableParallel({
        "scripts_docs": retriever_scripts,
        "pdf_docs": retriever_pdf,
        "web_docs": retriever_web_scraped
    })
    | (lambda docs: docs["scripts_docs"] + docs["pdf_docs"] + docs["web_docs"])
)

# --- Define RAG Prompt Template --- #
rag_prompt_template = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful AI assistant aboard Deep Space Nine. Use the provided context to answer the user's questions. If you don't know the answer, state that you don't have enough information."),
    ("human", "Context: {context}\n\nQuestion: {question}")
])

# --- Build RAG Chain --- #
rag_chain = (
    {
        "context": combined_retriever_runnable | (lambda docs: "\n\n".join([doc.page_content for doc in docs])),
        "question": RunnablePassthrough()
    }
    | rag_prompt_template
    | llm
    | StrOutputParser()
)

def predict_rag(message, history):
    """Invokes the RAG chain with the given question and returns the response."""
    response = rag_chain.invoke(message)
    return response

# --- Gradio UI with LCARS Styling --- #
lcars_css = """
.gradio-container {
    background-color: #000000 !important;
    font-family: "Antonio", "Arial Nan", sans-serif !important;
}
.lcars-header-pill {
    background-color: #ff9900;
    color: black;
    border-radius: 20px 0px 0px 20px;
    padding: 10px;
    font-weight: bold;
}
.lcars-sidebar {
    background-color: #cc6699;
    border-radius: 0px 0px 0px 20px;
    min-height: 300px;
}
#chatbot-window {
    border-left: 5px solid #ffcc00 !important;
    border-top: 5px solid #ffcc00 !important;
    padding-left: 15px;
}
"""

with gr.Blocks() as demo:
    with gr.Row():
        gr.HTML("<div class='lcars-header-pill'>LCARS SYSTEM TERMINAL v1.0</div>")
    with gr.Row():
        with gr.Column(scale=1, elem_classes="lcars-sidebar"):
            gr.Button("SYS STATUS", variant="primary")
            gr.Button("DATA INDEX", variant="secondary")
        with gr.Column(scale=4, elem_id="chatbot-window"):
            chatbot = gr.ChatInterface(fn=predict_rag)

demo.launch(css=lcars_css)
