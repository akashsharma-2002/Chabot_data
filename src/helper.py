from langchain_community.document_loaders import TextLoader, PyPDFLoader, DirectoryLoader
from langchain.text_splitter import CharacterTextSplitter, RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain_huggingface import HuggingFaceEmbeddings
# NEW: history imports (as requested)
from langchain_community.chat_message_histories import ChatMessageHistory  # in-memory impl
from langchain_core.chat_history import BaseChatMessageHistory              # interface
from langchain_core.runnables.history import RunnableWithMessageHistory     # wrapper
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder  # history slot if needed
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

#extracting text from pdf file
def extract_text_from_pdf(data):
    loader = DirectoryLoader(
        data, 
        glob="*.pdf", 
        loader_cls=PyPDFLoader
    )

    documents = loader.load()
    return documents


#trimming down the extracted contecxt only to what requiredp parameter
#we need only source and page_conetent from the extracted text

def trim_extracted_data(extract_text):
    """
    Triming the extracted data here , note is source is stored in metadata
    and page_content is stored directly in page_content attribute.

    Here i need to use Document class from langchain.schema to create new Document
    so whenever i trim out the data it should be kept inside document 
    class only.
    
    """
    trimmed_data=[]
    for i in extract_text:
        trimmed_data.append(Document(
            source=i.metadata.get("source"),
            page_content=i.page_content
        )
    )

    return trimmed_data
#creating chunks of data from the extracted text
def create_chunks(trimmed_extracted_doc):
    text_splitter=RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    docs=text_splitter.split_documents(trimmed_extracted_doc)
    return docs

#now embedding the chunks of data converting into embeddings

def embed_chunks():
    model_name="sentence-transformers/all-MiniLM-L6-v2"
    embeddings=HuggingFaceEmbeddings(model_name=model_name)
    return embeddings

store={}

def get_session_history(session_id: str) -> BaseChatMessageHistory:
    """Get the chat history for a given session ID, creating it if necessary."""
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    
    # Get the history
    history = store[session_id]
    
    # Keep only last 40 messages (20 Q&A pairs)
    if len(history.messages) > 40:
        history.messages = history.messages[-40:]  # Keep last 40 messages
    
    return history


