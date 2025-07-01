import os
from dotenv import load_dotenv
from langchain_community.document_loaders import DirectoryLoader, BSHTMLLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma

os.environ["CHROMA_TELEMETRY"] = "FALSE"

load_dotenv()

token = os.environ["OPENAI_API_KEY"]
endpoint = "https://models.inference.ai.azure.com"
model_name = "text-embedding-3-small"

loader = DirectoryLoader(
    "./articles",
    glob="**/*.html",
    loader_cls=BSHTMLLoader
)
raw_documents = loader.load()

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
    separators=["\n\n", "\n", " ", ""]
)
documents = text_splitter.split_documents(raw_documents)

embeddings = OpenAIEmbeddings(
    model=model_name,
    openai_api_key=token,
    openai_api_base=endpoint
)

persist_directory = "./chroma_db"
batch_size = 100
vectorstore = None

for i in range(0, len(documents), batch_size):
    batch = documents[i:i + batch_size]
    if vectorstore is None:
        vectorstore = Chroma.from_documents(
            batch,
            embedding=embeddings,
            persist_directory=persist_directory
        )
    else:
        vectorstore.add_documents(batch)

