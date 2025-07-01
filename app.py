import os
from dotenv import load_dotenv
import streamlit as st

from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain_openai import ChatOpenAI
from langchain.embeddings import OpenAIEmbeddings
import base64

# Load and encode your local image to base64
def get_base64_image(image_path):
    with open(image_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode()

# Image path
img_path = "./resources/image.jpg"
encoded_img = get_base64_image(img_path)

# Load .env
load_dotenv()

# Setup LLM
llm = ChatOpenAI(
    model="openai/gpt-4.1-mini",
    api_key=os.environ["OPENAI_API_KEY"],
    base_url="https://models.github.ai/inference"
)

# Embeddings
embedding_model = OpenAIEmbeddings(
    model="text-embedding-3-small",
    api_key=os.environ["OPENAI_API_KEY"],
    base_url="https://models.inference.ai.azure.com"
)

# Load Chroma vector store
persist_directory = "./chroma_db"
vectorstore = Chroma(
    embedding_function=embedding_model,
    persist_directory=persist_directory
)
retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

# Setup RetrievalQA chain
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever,
    chain_type="stuff"
)

# --- Streamlit UI Styling ---
st.set_page_config(page_title="Financial Stock Assistant", layout="wide")

st.markdown(f"""
    <style>
    /* Ensure container fills viewport */
    .stApp {{
        position: relative;
        min-height: 100vh;
        width: 100%;
        overflow: hidden;
        z-index: 0;
    }}

    /* Background image layer */
    .stApp::before {{
        content: "";
        position: fixed;  /* fixed to viewport */
        top: 0;
        left: 0;
        width: 100vw;
        height: 100vh;
        background-image: url("data:image/jpg;base64,{encoded_img}");
        background-size: cover;
        background-position: center center;
        background-repeat: no-repeat;
        z-index: -2;
        pointer-events: none;
    }}

    /* Black overlay layer */
    .stApp::after {{
        content: "";
        position: fixed;
        top: 0;
        left: 0;
        width: 100vw;
        height: 100vh;
        background-color: rgba(0, 0, 0, 0.75);
        z-index: -1;
        pointer-events: none;
    }}

    /* Your main content styling */
    .main {{
        background-color: transparent;
        padding: 1rem;
        position: relative;
        z-index: 1;
    }}


    h1.title {{
        margin-top: 0 !important;
        padding-top: 0 !important;
        font-size: 4.8rem;
        font-weight: 800;
        background: -webkit-linear-gradient(45deg,  #00d6ff, #ff4ecd);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        position: relative;
        z-index: 2;
    }}

    h2.subtitle {{
        font-size: 1.3rem;
        font-weight: 400;
        color: #ccc;
        text-align: center;
        margin-bottom: 1rem;
        position: relative;
        z-index: 2;
    }}

    .stSelectbox > label {{
        font-weight: 600;
        color: #00d6ff;
        position: relative;
        z-index: 2;
    }}

    /* Style the actual dropdown box */
    .stSelectbox div[data-baseweb="select"] > div {{
        border: 2px solid #00d6ff !important;
        border-radius: 8px !important;         
        background-color: #ff4ecd !important; 
        color: #003366 !important;             
        font-weight: 600 !important;
        font-size: 1rem !important;
    }}

    .stSelectbox div[data-baseweb="select"] > div > div {{
        color: #003366 !important;
    }}

    .stSelectbox div[data-baseweb="select"] ul {{
        background-color: #f0fbff !important;
        border-radius: 8px !important;
    }}

    .stSelectbox div[data-baseweb="select"] ul li:hover {{
        background-color: #00d6ff !important;
        color: white !important;
    }}
    
""", unsafe_allow_html=True)



# --- UI Content ---
st.markdown('<h1 class="title">Financial Stock Assistant</h1>', unsafe_allow_html=True)
st.markdown('<h2 class="subtitle">Generate Insightful, Data-Driven Stock Reports</h2>', unsafe_allow_html=True)
st.divider()

# Report Type
report_type = st.selectbox(
    "Select Report Type",
    ("Single Stock Outlook", "Competitor Analysis")
)

# --- Single Stock ---
if report_type == "Single Stock Outlook":
    symbol = st.selectbox(
        "Select Stock Symbol",
        ("", "AAPL", "MSFT", "GOOGL", "TSLA", "TSM", "META", "NVDA"),
        index=0
    )

    if symbol:
        with st.spinner(f"🔍 Generating report for {symbol}..."):
            query = f"Write a report on the outlook for {symbol} stock from the years 2023-2027. Include potential risks and headwinds. Just give the report only"
            response = qa_chain.run(query)

        # --- Styled output box ---
        st.markdown(f"<div class='report-box'><h3>📈 Report for {symbol}</h3><p>{response}</p></div>", unsafe_allow_html=True)

# --- Competitor Analysis ---
elif report_type == "Competitor Analysis":
    col1, col2 = st.columns(2)
    with col1:
        symbol1 = st.selectbox(
            "Select First Stock Symbol",
            ("", "AAPL", "MSFT", "GOOGL", "TSLA", "TSM", "META", "NVDA"),
            index=0,
            key="symbol1"
        )
    with col2:
        symbol2 = st.selectbox(
            "SelectSecond Stock Symbol",
            ("", "AAPL", "MSFT", "GOOGL", "TSLA", "TSM", "META", "NVDA"),
            index=0,
            key="symbol2"
        )

    if symbol1 == symbol2 and symbol1 != "":
        st.error("❌ Please select two different stock symbols.")
    elif symbol1 and symbol2:
        with st.spinner(f"🔍 Generating comparative report for {symbol1} and {symbol2}..."):
            query = f"Write a comparative report between {symbol1} and {symbol2} stocks and their market positions. Just give the report only"
            response = qa_chain.run(query)

        # --- Styled output box ---
        st.markdown(f"<div class='report-box'><h3>🆚 {symbol1} vs {symbol2}</h3><p>{response}</p></div>", unsafe_allow_html=True)