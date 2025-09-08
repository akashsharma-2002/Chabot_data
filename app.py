from flask import Flask,render_template,request,jsonify
from src.helper import embed_chunks,extract_text_from_pdf,trim_extracted_data,create_chunks,get_session_history
from langchain_community.chat_message_histories import ChatMessageHistory  # in-memory impl
from langchain_core.chat_history import BaseChatMessageHistory              # interface
from langchain_core.runnables.history import RunnableWithMessageHistory     # wrapper
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder  # history slot if needed
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_pinecone import PineconeVectorStore
from langchain_openai import ChatOpenAI
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from pinecone import Pinecone
from pinecone import ServerlessSpec
from dotenv import load_dotenv
import os
from src.prompt import *

load_dotenv()
app=Flask(__name__)

PINECONE_API_KEY=os.getenv("PINECONE_API_KEY")
LLM_API_KEY=os.getenv("LLM_API_KEY")

os.environ["PINECONE_API_KEY"]=PINECONE_API_KEY
os.environ["LLM_API_KEY"]=LLM_API_KEY

embeddings=embed_chunks()
index_name="data-chatbot"

docsearch = PineconeVectorStore.from_existing_index(
    index_name=index_name,
    embedding=embeddings
    
)

#creating a retriever from the docsearch
retriever=docsearch.as_retriever(search_type="similarity", search_kwargs={"k":3}
                                )

#importing the github free model for chatbot

#importing the github free model for chatbot

chatModel = ChatOpenAI(
    model="openai/gpt-4.1",              
    api_key=os.environ["LLM_API_KEY"],
    base_url="https://models.github.ai/inference"
)

prompt = ChatPromptTemplate.from_messages([
    ("system",system_prompt),
    MessagesPlaceholder("chat_history"),  # <-- # THIS needs to be added (history slot)
    ("human", "{input}")
])

question_answering_chain=create_stuff_documents_chain(chatModel,prompt)
rag_chain=create_retrieval_chain(retriever,question_answering_chain)

conversational_rag_chain = RunnableWithMessageHistory(
    rag_chain, 
    get_session_history,
    input_messages_key="input",
    history_messages_key="chat_history",
    output_messages_key="answer")

@app.route("/")
def index():
    return render_template("chat.html")    

@app.route("/get",methods=["GET","POST"])
def chat():
    # Use a unique session ID per browser tab (or use a default one)
    sid = request.form.get("session_id", "demo-001") #here the session_id is main if not then fall back to demo-001
    msg = request.form["msg"]
    print(f"Session: {sid}, Input: {msg}")
    
    response = conversational_rag_chain.invoke(
        {"input": msg},
        config={"configurable": {"session_id": sid}}
    )
    print(f"Response: {response.get('answer')}")
    return str(response.get('answer', 'Sorry, I could not process your request.'))



if __name__=="__main__":
    import os
    # Render uses PORT environment variable, default to 10000 for Render
    port = int(os.environ.get('PORT', 10000))
    app.run(host="0.0.0.0", port=port, debug=False)
