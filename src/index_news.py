import os
from dotenv import load_dotenv
from llama_index import GPTVectorStoreIndex, SimpleDirectoryReader, ServiceContext
from langchain_community.embeddings import OpenAIEmbeddings

load_dotenv()

token = os.environ["OPENAI_API_KEY"]
endpoint = "https://models.inference.ai.azure.com"
model_name = "text-embedding-3-small"

embeddings = OpenAIEmbeddings(
    model = model_name,
    openai_api_key = token,
    openai_api_base = endpoint,
)
service_context = ServiceContext.from_defaults(embed_model=embeddings)

documents = SimpleDirectoryReader("./articles").load_data()

index = GPTVectorStoreIndex.from_documents(documents, service_context=service_context)
index.storage_context.persist()