
from dotenv import load_dotenv
import os
from pinecone import Pinecone
from langchain_pinecone import PineconeVectorStore
from pinecone import ServerlessSpec
from src.helper import extract_text_from_pdf,trim_extracted_data,create_chunks,embed_chunks 
load_dotenv()

PINECONE_API_KEY=os.getenv("PINECONE_API_KEY")
LLM_API_KEY=os.getenv("LLM_API_KEY")

os.environ["PINECONE_API_KEY"]=PINECONE_API_KEY
os.environ["LLM_API_KEY"]=LLM_API_KEY


extracted_data=extract_text_from_pdf("data")
trimmed_extracted_doc=trim_extracted_data(extracted_data)
chunks_text=create_chunks(trimmed_extracted_doc)
embeddings=embed_chunks(chunks_text)

pinecone_api_key=PINECONE_API_KEY
pc= Pinecone(api_key=pinecone_api_key)

#pinecode index creation for pinecode and storing the embeddings inside it

index_name="data-chatbot"
if not pc.has_index(index_name):
    pc.create_index(
        name=index_name,
        dimension=384,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1")
    )   
index = pc.Index(index_name)

# storing inside the index created


docsearch = PineconeVectorStore.from_documents(
    documents=chunks_text,
    embedding=embeddings,
    index_name=index_name
)