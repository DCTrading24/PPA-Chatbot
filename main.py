import streamlit as st  # Web app framework
import os  # File path operations
import openai  # OpenAI API for generating responses
from langchain.document_loaders import PyPDFLoader  # PDF Loader for extracting text
from langchain.text_splitter import CharacterTextSplitter  # Splits text into chunks
from langchain.embeddings import OpenAIEmbeddings  # Embedding model for vector search
from langchain.vectorstores import Chroma  # Chroma Vector Store for retrieval
#import chromadb  # ChromaDB for vector storage
from dotenv import load_dotenv  # Load environment variables
#from chromadb.api.fastapi import FastAPI

# Define a function to add a background color only if there is content
def add_background_for_text(content):
    if content:  # Only apply the background if there's content
        return f'<div style="background-color: rgba(255, 255, 255, 0.6); padding: 15px; border-radius: 10px;">{content}</div>'
    return content  # If no content, just return it as is

# --- Load API keys ---
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY") # to run it locally
# api_key = st.secrets["OPENAI_API_KEY"] # to run it on streamlit
openai.api_key = api_key

# --- Streamlit UI Configuration ---
st.set_page_config(page_title="PPA Explainer - AI Chatbot", page_icon="🤖", layout="wide")

# --- Header with Logo & Title ---
st.markdown("""
    <style>
        .stApp {
            background: url("https://img.freepik.com/premium-photo/wind-turbines-green-field-sunset-promote-renewable-energy-sustainable-living_38013-10280.jpg") no-repeat center center fixed;
            background-size: cover;
            padding: 0;
        }

        .stImage {
            z-index: 1;
        }
    </style>
""", unsafe_allow_html=True)


st.markdown("""
    <div class='title-container' style='text-align: center;'>
        <h1>🤯 PPA Explainer - AI Chatbot</h1>
        <p>Your AI-powered assistant for insights in the PPA Business</p>
    </div>
""", unsafe_allow_html=True)

st.markdown("""<br><br>""", unsafe_allow_html=True)  # Spacer

# --- Initialize Chat History ---
if "messages" not in st.session_state:
    st.session_state["messages"] = []

# --- Display Chat History ---
for message in st.session_state["messages"]:
    with st.chat_message(message["role"]):
        content_with_background = add_background_for_text(message["content"])
        st.markdown(content_with_background, unsafe_allow_html=True)

# --- User Input ---
user_question = st.chat_input("Ask me anything...")
embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
db = Chroma(
    persist_directory="./chroma_db",
    embedding_function=embeddings
)

if user_question:
    st.session_state["messages"].append({"role": "user", "content": user_question})
    with st.chat_message("user"):
        question_with_background = add_background_for_text(user_question)
        st.markdown(question_with_background, unsafe_allow_html=True)
    
    # --- Processing Animation ---
    st.spinner("Thinking... 🤥🤔")

    retrieved_docs = db.similarity_search(str(user_question), k=10)

    if retrieved_docs:
            source_info = "\n".join(
            [f"📖 **Source:** {os.path.basename(doc.metadata.get('source', 'Unknown'))}, Page: {doc.metadata.get('page', 'N/A')}" for doc in retrieved_docs]
            )
            context_text = retrieved_docs[0].page_content
    else:
            source_info = "❌ **No relevant information found.**"
            context_text = "No context available."

        # --- Construct AI Prompt ---

    prompt = f"""
## SYSTEM ROLE
You are a **highly knowledgeable, factual, and precise AI assistant** specializing in **renewable power markets**, with a focus on **Power Purchase Agreements**.  
Your answers must be based **exclusively** on the content provided from technical books and understandable for humans not familiar with the topic.  

🚨 **IMPORTANT:**  
- If the answer **cannot be found**, clearly state:  
  👉 *"The provided context does not contain this information."*  
- **DO NOT** provide personal opinions, external knowledge, or speculation.  

---

## 🟢 USER QUESTION  
The user has asked:  
**"{user_question}"**  

---

## 📚 CONTEXT (Relevant Extracts from Technical Books)  
"{context_text}"  

---

## 🛠️ GUIDELINES FOR ANSWERING  

### ✅ 1. **Accuracy & Context-Adherence**  
- Use **only** the information provided in `CONTEXT`.  
- **If the information is missing, say so.** Do NOT generate an answer from external knowledge.  
- If applicable, cross-reference multiple sections from the book.  

### ✅ 2. **Structured & Clear Response**  
Your answer should follow this format:  

    **Brief, Clear Title**
    (Short defintion and overview) 

    **🔹 Key Points:**  
    - (Bullet-point summary of key background information and considerations)  
    - (Additional relevant insights)  
    - (If suitable for the question formulate creative suggestions for strategies)

    **📖 Source(s):**  
    - *[Book Title], Page(s): [...]*
    - If page numbers are missing, state:  
    👉 *"The provided context does not specify the page number, but the book title is [Book Name]."*
    - If no sources are found, state:  
    👉 *"No sources available in the provided context."*  


### ✅ 3. **Transparency & Source Referencing**  
- Every answer **must** reference **the book title & page numbers**.  
- If multiple sources are available, summarize **consistently** across them.  

---

## 🎯 TASK  
1. Provide a **direct, structured, and clear response**.  
2. **If the information is missing, state it explicitly.**  
3. Format the response **in Markdown** for readability.  

---

🔹 **Now generate the answer using these instructions.**  
"""



        # --- Call OpenAI GPT ---
    client = openai.OpenAI()
    model_params = {
            'model': 'gpt-4o', # 
            'temperature': 1, # the higher, the more creative - temperature is a hyperparameter that controls the randomness of predictions
            'max_tokens': 4000, # token limit for the response
            'top_p': 0.9,
            'frequency_penalty': 0.5, #the higher, the more creative
            'presence_penalty': 0.6 #the higher, the more creative, presence_penalty is a hyperparameter that penalizes new tokens based on their presence in the context
}

    messages = [{'role': 'user', 'content': prompt}]
    completion = client.chat.completions.create(messages=messages, **model_params, timeout=120)
    answer = completion.choices[0].message.content

    # --- Display AI Response ---
    st.session_state["messages"].append({"role": "assistant", "content": answer})
    with st.chat_message("assistant"):
        answer_with_background = add_background_for_text(answer)
        st.markdown(answer_with_background, unsafe_allow_html=True)