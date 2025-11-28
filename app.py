import os
from dotenv import load_dotenv
from PIL import Image
import streamlit as st
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

# Load environment variables
load_dotenv()

# API Key Validation
if not os.getenv("GOOGLE_API_KEY"):
    st.error("⚠️ Google API Key not found! Please create a .env file containing GOOGLE_API_KEY=your_key")
    st.stop()

DB_FAISS_PATH = "vectorstore/db_faiss"

# Load the logo safely
try:
    im = Image.open("assets/factorial24_logo.ico")
except FileNotFoundError:
    # Fallback if logo missing, though not ideal for production
    im = None 

st.set_page_config(
    page_title = "Factorial24 OnboardIQ",
    page_icon = im,
    layout = "wide",
    initial_sidebar_state = "expanded"
)

# --- CSS STYLING (Unchanged) ---
st.markdown("""
<style>
    /* Main Header Styling */
    .main-header {
        font-size: 2.5rem;
        color: #4A90E2; 
        font-weight: 700;
        margin-bottom: 0px;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #666;
        margin-bottom: 20px;
    }
    /* Chat Message Styling */
    .stChatMessage {
        border-radius: 10px;
        padding: 10px;
    }
    /* Hide Default Streamlit Menu */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
</style>
""", unsafe_allow_html = True)

# --- HELPER FUNCTIONS ---

def is_personal_query(query):
    """
    Detects if a query is personal/conversational directed at the bot 
    (e.g., identity, age, location) rather than the knowledge base.
    """
    query = query.lower().strip()
    
    # List of triggers that indicate a personal question about the bot
    personal_triggers = [
        "who are you", "what are you", "your name", "your age", 
        "how old", "where do you live", "where are you", 
        "tell me about yourself", "are you human", "are you ai",
        "who created you", "what can you do", "hello", "hi", "hey"
    ]
    
    # Check if any trigger phrase appears in the query
    return any(trigger in query for trigger in personal_triggers)

@st.cache_resource
def load_knowledge_base():
    """
    Loads the Vector DB and initializes the Retrieval Chain.
    Cached to prevent reloading on every interaction.
    """
    
    embeddings = HuggingFaceEmbeddings(
        model_name = 'sentence-transformers/all-MiniLM-L6-v2',
        model_kwargs = {'device': 'cpu'}
    )
    
    try:
        db = FAISS.load_local(DB_FAISS_PATH, embeddings, allow_dangerous_deserialization = True)
    except Exception as e:
        st.error(f"⚠️ Critical Error: Could not load the knowledge base. \nDetails: {e}")
        return None

    # 'k': 3 means we fetch the top 3 most relevant chunks
    retriever = db.as_retriever(search_kwargs = {'k': 3})

    llm = ChatGoogleGenerativeAI(
        model = "gemini-pro-latest", 
        temperature = 0.3,
        convert_system_message_to_human = True
    )

    # UPDATED Custom Persona Prompt to handle Personal Questions vs Context
    custom_prompt_template = """
    You are 'OnboardIQ', the dedicated AI Assistant for Factorial24.
    
    Instructions:
    1. IDENTITY & CHITCHAT: If the user asks about you (e.g., "Who are you?", "What do you do?", "How are you?"), answer naturally as OnboardIQ. You do NOT need the context for this.
    2. KNOWLEDGE BASE: For questions about Factorial24, policies, projects, or teams, answer STRICTLY based on the provided context below.
    3. UNKNOWN: If the answer is not in the context and it is not a question about your identity, politely state that you don't have that information.

    Context: {context}

    Question: {question}

    Answer:
    """
    prompt = PromptTemplate(template = custom_prompt_template, input_variables = ['context', 'question'])

    qa_chain = RetrievalQA.from_chain_type(
        llm = llm,
        chain_type = "stuff",
        retriever = retriever,
        return_source_documents = True, 
        chain_type_kwargs = {'prompt': prompt}
    )
    
    return qa_chain


# --- SIDEBAR UI (Unchanged) ---
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/4712/4712035.png", width=80)
    st.title("Factorial24")
    st.markdown("### OnboardIQ Assistant")
    st.markdown("---")
    
    st.markdown("### 💡 Quick Questions")
    st.info(
        "• What is the leave policy?\n"
        "\n• Who is the manager for Project Alpha?\n"
        "\n• What is the tech stack?\n"
        "\n• Do I need to come to the office on Fridays?"
    )
    
    st.markdown("---")
    st.markdown("**Capabilities:**")
    st.markdown("✅ HR Policies & Leave")
    st.markdown("✅ Project Alpha Specs")
    st.markdown("✅ Team Hierarchy")
    st.markdown("---")
    
    if st.button("🗑️ Clear Conversation", use_container_width = True):
        st.session_state.messages = []
        st.rerun()


# --- MAIN CHAT INTERFACE ---
st.markdown('<div class="main-header">🚀 OnboardIQ</div>', unsafe_allow_html = True)
st.markdown('<div class="sub-header">Your AI Companion for navigating Factorial24</div>', unsafe_allow_html = True)

# Initialize Chat History
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display Chat History
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# User Input Handling
if prompt := st.chat_input("Ask a question about Factorial24..."):
    # 1. Display User Message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # 2. Process Response
    chain = load_knowledge_base()
    
    # 3. Detect if this is a personal question
    is_personal = is_personal_query(prompt)
    
    if chain:
        with st.chat_message("assistant"):
            with st.spinner("🔍 Consulting the organizational knowledge base..."):
                try:
                    # Run the chain
                    result = chain.invoke({"query": prompt})
                    response_text = result['result']
                    source_docs = result['source_documents']
                    
                    # Display Answer
                    st.markdown(response_text)
                    
                    # Display Citations (The "Wow" Factor) - CONDITIONAL LOGIC
                    # Only show docs if they exist AND it is NOT a personal question
                    if source_docs and not is_personal:
                        with st.expander("📚 Reference Documents (Sources)"):
                            for i, doc in enumerate(source_docs):
                                # Clean filename
                                raw_source = doc.metadata.get('source', 'Unknown')
                                filename = os.path.basename(raw_source)
                                
                                st.markdown(f"**Source {i+1}:** `{filename}`")
                                st.caption(f"\"{doc.page_content[:150]}...\"") # Preview snippet
                                st.divider()
                    
                    # Save Assistant Message
                    st.session_state.messages.append({"role": "assistant", "content": response_text})
                    
                except Exception as e:
                    st.error(f"An error occurred: {str(e)}")