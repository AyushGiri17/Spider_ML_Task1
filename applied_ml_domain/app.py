# =========================================================

# IMPORTS

# ==========================================================

# os:

# Used to check whether a FAISS vector database already exists

# on disk. If it exists, we load it instead of recreating it.

# dotenv:

# Loads environment variables such as the Groq API key from

# the .env file to avoid hardcoding secrets inside the code.

# HuggingFaceEmbeddings:

# Converts text chunks into dense vector representations that

# can be stored and searched using semantic similarity.

# ChatGroq:

# Connects the application to Groq-hosted LLMs which generate

# final answers from the retrieved document context.

# RecursiveCharacterTextSplitter:

# Splits large PDF documents into smaller chunks that fit

# within the context window of embedding and language models.

# DirectoryLoader + PyPDFLoader:

# Loads all PDF files from the papers folder and extracts text.

# FAISS:

# Vector database used to store embeddings and perform

# similarity search during retrieval.

# Streamlit:

# Creates the web interface for interacting with the chatbot.

# LangChain Runnables:

# Used to build the RAG pipeline by connecting retrieval,

# prompt creation and LLM generation steps together.


import os

from dotenv import load_dotenv

from langchain_huggingface import (HuggingFaceEmbeddings)
from langchain_groq import (ChatGroq)

from langchain_text_splitters import RecursiveCharacterTextSplitter

from langchain_community.document_loaders import ( DirectoryLoader,PyPDFLoader)

from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
import streamlit as st

from langchain_core.runnables import (RunnableParallel,RunnablePassthrough,RunnableLambda)

from langchain_core.output_parsers import (StrOutputParser)
load_dotenv()

#======================================================================
#STEP 1: DOCUMENT LOADING
#All PDF research papers inside the "papers" folder are
#automatically loaded and converted into LangChain documents
#======================================================================
st.session_state.setdefault("chat_history", [])


loader = DirectoryLoader("papers",glob="*.pdf",loader_cls=PyPDFLoader)
documents = loader.load()

#=========================================================================================	
#Step 2 - Indexing (Text Splitting)

#earch papers can be very large. Therefore documents are
#split into smaller chunks before embedding generation.
#chunk_size = 1000
#Large enough to preserve useful context while remaining
#small enough for efficient retrieval.
#chunk_overlap = 200
#Maintains continuity between neighboring chunks and reduces
#information loss at chunk boundaries.
#===================================================================================================
splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
chunks = splitter.split_documents(documents)





# ==========================================================

# STEP 3: EMBEDDING GENERATION & VECTOR STORAGE

# ==========================================================

# all-MiniLM-L6-v2 is a lightweight and widely-used sentence

# transformer model suitable for semantic search tasks.

# Each chunk is converted into a numerical vector and stored

# inside a FAISS vector database.

# If a FAISS database already exists, it is loaded directly.

# This avoids recomputing embeddings every time the app starts,

# significantly reducing startup time.
#====================================================================

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

if os.path.exists("faiss_db"):

    vector_store = FAISS.load_local("faiss_db",embeddings, allow_dangerous_deserialization=True)

else:

    vector_store = FAISS.from_documents(chunks,embeddings)

    vector_store.save_local("faiss_db")
#==================================================================================================
#Step 4 - Retrieval k = 6 to retrieve top 6 similiar chunks
#===========================================================================================================

retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 6})
#===========================================================================================================

#Step 5  - Augmentation , 1500 token size is considered good for summary
#based answers between multiple pdf.
#Here context question and history all goes down as combined prompt
#===========================================================================================================

llm = ChatGroq(model="llama-3.3-70b-versatile",temperature=0.2,max_tokens=1500)
prompt = PromptTemplate(template="""Chat History:{history}
Context:{context}
Question:{question}
Answer ONLY from the provided context.If the answer cannot be found,say "I don't know."
""",
    input_variables=["history","context","question"])
#===========================================================================================================

#step6  Building a Chain , as parallel chain was required, using runnableParallel 
#and runnable lamda 
#as question is required in both semantic search and llm pass through is used
#===========================================================================================================

def format_docs(retrieved_docs): # to get retrieved documents
  context_text = "\n\n".join(doc.page_content for doc in retrieved_docs)
  return context_text
  
  
def format_chat_history():

    history = ""

    for chat in st.session_state.get("chat_history", []):

        history += f"User: {chat['question']}\n"
        history += f"Assistant: {chat['answer']}\n\n"

    return history
	
def extract_sources(retrieved_docs):

    sources = []

    for doc in retrieved_docs:

        source = doc.metadata.get("source", "Unknown")
        page = doc.metadata.get("page","Unknown")
        sources.append(f"{source} (page {page})")


    return list(set(sources))

  
parallel_chain = RunnableParallel({
    "context": retriever | RunnableLambda(format_docs),
    "question": RunnablePassthrough(),

    "history": RunnableLambda(lambda x: format_chat_history())})
       
parser = StrOutputParser()
main_chain = parallel_chain | prompt | llm | parser
#===========================================================================================================

# Step 7 Streamlit UI interface

st.title("Research Paper Assistant")

# Display previous chat history

for chat in st.session_state.chat_history:

    with st.chat_message("user"):
        st.write(chat["question"])

    with st.chat_message("assistant"):
        st.write(chat["answer"])


# Chat input box at bottom

question = st.chat_input("Ask a question about your papers...")
    
if question:

    with st.chat_message("user"):
        st.write(question)
    #putting question in retriever to get related chunks
    retrieved_docs = retriever.invoke( question )
    #to get the origin sources of chunks from metadata

    sources = extract_sources(retrieved_docs)
        
    # to invoke main chain and get final response

    response = main_chain.invoke(question)
    # to append chat history

    st.session_state.chat_history.append({ "question": question,"answer": response})
       
    # Display the chatbot response inside an assistant-style
    # chat message. Source references used during retrieval are
    # shown inside a collapsible dropdown to keep the interface
    # clean while maintaining transparency.

    with st.chat_message("assistant"):

        st.write(response)

        with st.expander("Sources"):

            for source in sources:
                st.write(source)