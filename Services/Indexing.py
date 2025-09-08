from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters.character import CharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from pinecone import Pinecone, ServerlessSpec
from Services.Credentials import GEMINI_API_KEY as GemKey # importing GEMINI API KEY
from Services.Credentials import PINECONE_API_KEY as PineKey # importing PINECONE API KEY
import numpy as np

PINECONE_API_KEY = PineKey
PINECONE_ENV = "us-east1-gcp"
PINECONE_INDEX = "gemini-index"

#Loading the document
loader_pdf = PyPDFLoader(r"C:\Users\Banwa\Desktop\CHITKARA\SEM-1\DICE\GOOGLE CLOUD\v0\PDF\A Clinical Guide Manual.pdf")
pages_pdf = loader_pdf.load()

#Removing the newline characters and some noise and splitting the content into chunks
for i in range(len(pages_pdf)):
    pages_pdf[i].page_content = ' '.join(pages_pdf[i].page_content.split())
# print("First chunk sample:", pages_character_split[0].page_content[:200]) #printing a test chunk

char_splitter = CharacterTextSplitter(
    separator=".",
    chunk_size=500,
    chunk_overlap=10
)
pages_character_split = char_splitter.split_documents(pages_pdf)
# print("Number of chunks:", len(pages_character_split))

# Embedding the chunks and making the vectors !
embedding = GoogleGenerativeAIEmbeddings(
    model="models/embedding-001",   # Gemini embedding endpoint
    google_api_key=GemKey  # Or set via env var GOOGLE_API_KEY
)

#Initialising Pinecone
pc = Pinecone(api_key=PineKey)  # replace with your key

# Ensure index exists
if "gemini-index" not in pc.list_indexes().names():
    pc.create_index(
        name="gemini-index",
        dimension=768,        # Gemini embeddings dimension
        metric="cosine",      # similarity metric
        spec=ServerlessSpec(cloud="aws", region="us-east-1")

    )

# Connect to index
index = pc.Index("gemini-index")

batch_size = 32
batch = []

for i, doc in enumerate(pages_character_split):
    vector = embedding.embed_query(doc.page_content)
    metadata = {
        "source": "A Clinical Guide Manual.pdf",
        "chunk_index": i,
        "text_snippet": doc.page_content[:200]
    }
    uid = f"chunk-{i}"
    batch.append((uid, vector, metadata))

    if len(batch) >= batch_size:
        index.upsert(vectors=batch)
        batch = []

if batch:
    index.upsert(vectors=batch)

print(f"Uploaded {len(pages_character_split)} chunks to Pinecone index 'gemini-index'")
