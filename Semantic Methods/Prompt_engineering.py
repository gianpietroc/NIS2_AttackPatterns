from typing import TypedDict, List
import os
import pickle
import hashlib
from langchain_openai import ChatOpenAI

# Langchain imports
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.document_loaders import CSVLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.llms import Ollama
from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.prompts import PromptTemplate
from langchain.schema import Document
from langchain.prompts import FewShotPromptTemplate
from langchain_community.cache import InMemoryCache
import langchain
# LangGraph imports
from langgraph.graph import StateGraph, START, END

# Enable LangChain caching
langchain.llm_cache = InMemoryCache()

# First, verify files exist to fail early with clear errors
TEXT_PATH = "RAG_docs/results.txt"
CSV_PATH = "RAG_docs/Capec_part.csv"
CACHE_DIR = "cache_capec"
TEXT_CACHE_PATH = os.path.join(CACHE_DIR, "text_vector_store.pkl")
CSV_CACHE_PATH = os.path.join(CACHE_DIR, "csv_vector_store.pkl")

# Create cache directory if it doesn't exist
os.makedirs(CACHE_DIR, exist_ok=True)

# Check if files exist
if not os.path.exists(TEXT_PATH):
    raise FileNotFoundError(f"Text file not found: {TEXT_PATH}")
if not os.path.exists(CSV_PATH):
    raise FileNotFoundError(f"CSV file not found: {CSV_PATH}")

# Initialize LLM & Embeddings with consistent model
MODEL_NAME = "gpt-4o-mini"
print(f"Initializing Chat with model: {MODEL_NAME}")
#llm = Ollama(model=MODEL_NAME, temperature=0)
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(
    model=MODEL_NAME,
    temperature=0,
    max_tokens=None,
    timeout=None,
    max_retries=2,
    api_key="" #to add key
)

'''embeddings = OpenAIEmbeddings(
    model="text-embedding-ada-002",
    api_key="",   #to add key 
    )
'''
embeddings = OllamaEmbeddings(model="llama3.2:1b")

# Define State for LangGraph
class AgentState(TypedDict):
    question: str
    retrieved_texts: List[Document]
    csv_analysis: str
    final_answer: str

# Function to check if cache is valid
def is_cache_valid(cache_path, source_path):
    """Check if cache exists and is newer than source file"""
    if not os.path.exists(cache_path):
        return False
    
    cache_mtime = os.path.getmtime(cache_path)
    source_mtime = os.path.getmtime(source_path)
    
    return cache_mtime > source_mtime

# Load Text Data - Create vector store from text file
def load_text_data():
    print(f"Loading text data from {TEXT_PATH}")
    
    # Check if we can use cached vector store
    if is_cache_valid(TEXT_CACHE_PATH, TEXT_PATH):
        print("Using cached text vector store")
        with open(TEXT_CACHE_PATH, 'rb') as f:
            vector_store = pickle.load(f)
            vector_store = FAISS.deserialize_from_bytes(vector_store, embeddings, allow_dangerous_deserialization=True)
        return vector_store.as_retriever(search_kwargs={"k": 100})
    
    # If no valid cache, create new vector store
    print("Creating new text vector store")
    loader = TextLoader(TEXT_PATH)
    documents = loader.load()
    
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=3096, chunk_overlap=100)
    doc_chunks = text_splitter.split_documents(documents)
    print(f"Created {len(doc_chunks)} text chunks")

    vector_store = FAISS.from_documents(doc_chunks, embeddings, normalize_L2=True)
    
    # Save to cache
    with open(TEXT_CACHE_PATH, 'wb') as f:
        pickle.dump(FAISS.serialize_to_bytes(vector_store), f)
    
    return vector_store.as_retriever(search_kwargs={"k": 100})

# Load CSV Data - Create vector store from CSV
def load_csv_data():
    print(f"Loading CSV data from {CSV_PATH}")
    
    # Check if we can use cached vector store
    if is_cache_valid(CSV_CACHE_PATH, CSV_PATH):
        print("Using cached CSV vector store")
        with open(CSV_CACHE_PATH, 'rb') as f:
            vector_store = pickle.load(f)
            vector_store = FAISS.deserialize_from_bytes(vector_store, embeddings, allow_dangerous_deserialization=True)
        return vector_store.as_retriever(search_kwargs={"k": 600})
    
    # If no valid cache, create new vector store
    print("Creating new CSV vector store")
    loader = CSVLoader(
        file_path=CSV_PATH,
            encoding="utf8",
            csv_args={
                'delimiter':",",
                'skipinitialspace': True

        }
    )
    
    documents = loader.load()
    print(f"Loaded {len(documents)} documents from CSV")
    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=4000,
        chunk_overlap=200,
        separators=["\n\n", "\n", " ", ""]
    )
    
    doc_chunks = text_splitter.split_documents(documents)
    print(f"Created {len(doc_chunks)} CSV chunks")
    
    vector_store = FAISS.from_documents(doc_chunks, embeddings)
    
    # Save to cache
    with open(CSV_CACHE_PATH, 'wb') as f:
        pickle.dump(FAISS.serialize_to_bytes(vector_store), f)
    
    return vector_store.as_retriever(search_kwargs={"k": 600})

# Initialize retrievers once
print("Initializing retrievers...")
text_retriever = load_text_data()
csv_retriever = load_csv_data()
print("Retrievers initialized")

# Create a hash for caching function results
def create_hash(data):
    """Create a hash from a string or dictionary for caching purposes"""
    if isinstance(data, dict):
        data = str(sorted(data.items()))
    return hashlib.md5(data.encode()).hexdigest()

# LRU cache for text retrieval results
retrieval_cache = {}
def cached_retrieve_text(question):
    """Cached version of text retrieval"""
    cache_key = create_hash(question)
    if cache_key in retrieval_cache:
        print(f"Using cached text retrieval for question: {question}")
        return retrieval_cache[cache_key]
    
    print(f"Retrieving text documents for question: {question}")
    retrieved_docs = text_retriever.invoke(question)
    print(f"Retrieved {len(retrieved_docs)} text documents")
    
    # Cache the result
    retrieval_cache[cache_key] = retrieved_docs
    return retrieved_docs

# LRU cache for CSV retrieval results
csv_retrieval_cache = {}
def cached_retrieve_csv(question):
    """Cached version of CSV retrieval"""
    cache_key = create_hash(question)
    if cache_key in csv_retrieval_cache:
        print(f"Using cached CSV retrieval for question: {question}")
        return csv_retrieval_cache[cache_key]
    
    print(f"Retrieving CSV documents for question: {question}")
    retrieved_docs = csv_retriever.invoke(question)

    # Cache the result
    csv_retrieval_cache[cache_key] = retrieved_docs
    return retrieved_docs

# Step 1: Retrieve Relevant Text Data
def retrieve_text(state: AgentState):
    question = state["question"]
    retrieved_docs = cached_retrieve_text(question)
    return {"retrieved_texts": retrieved_docs}

# LRU cache for CSV analysis results
analysis_cache = {}
# Step 2: Analyze CSV Data
def analyze_csv(state: AgentState):
    question = state["question"]
    print(f"Analyzing CSV data for question: {question}")
    
    # Check if we have a cached analysis
    cache_key = create_hash(question)
    if cache_key in analysis_cache:
        print(f"Using cached CSV analysis for question: {question}")
        return {"csv_analysis": analysis_cache[cache_key]}
    
    try:
        # Get relevant attack techniques
        retrieved_docs = cached_retrieve_csv(question)
        
        # Create context from retrieved documents
        docs_content = "\n\n".join(doc.page_content for doc in retrieved_docs)
        #print("---", docs_content)

        # Define prompt for structuring the retrieved attack information
        prompt_template = """
        You are an AI assistant specializing in Attack Patterns. 
        Extract the Attack Patterns from the retrieved text, from the Retrieved Context {context}. 
        Give the output in the following format:

        **Attack Pattern ID**: <ID>  
        **Name**: <Name>  
        **Description**: <Description>
        """
        
        # Correct initialization with input_variables
        prompt = PromptTemplate(template=prompt_template, input_variables=["context"])
        formatted_prompt = prompt.format(context=docs_content)
        
        # Generate structured response - Handle different response formats
        response = llm.invoke(formatted_prompt)
        
        # Extract response content
        if isinstance(response, str):
            result = response
        elif hasattr(response, 'content'):
            result = response.content
        else:
            result = str(response)
        
        # Cache the result
        analysis_cache[cache_key] = result
        
        return {"csv_analysis": result}
        
    except Exception as e:
        print(f"Error in CSV analysis: {str(e)}")
        return {"csv_analysis": f"Error retrieving attack information: {str(e)}"}

# Cache for final answer generation
final_answer_cache = {}
# Step 3: Generate the Final Answer
def generate_final_answer(state: AgentState):
    print("Generating final answer")
    
    # Create a cache key from the state's relevant parts
    '''cache_key = create_hash({
        'docs': "\n".join(doc.page_content for doc in state["retrieved_texts"]),
        'csv_analysis': state["csv_analysis"]
    })'''
    
    # Check cache
    '''if cache_key in final_answer_cache:
        print("Using cached final answer")
        return {"final_answer": final_answer_cache[cache_key]}'''
    
    # Extract content from retrieved documents
    docs_content = "\n".join(doc.page_content for doc in state["retrieved_texts"])
    print("---DOCS",docs_content)
    csv_analysis = state["csv_analysis"]

    print("Docs content length:", len(docs_content))
    print("CSV analysis length:", len(csv_analysis))
    
    # Define example data for few-shot prompting
    examples = [
        {
            "class": "MemberState",
            "action": "ensure",
            "object": "MemoryControls",
            "attack_id": "100",
            "attack_name": "Buffer Overflow",
            "description": "Buffer Overflow targets the lack of Memory controls",
            "explanation": "The object 'MemoryControls' directly relates to the Buffer Overflow attack, as the attack exploits insufficient memory protections."
        },
        {
            "class": "System",
            "action": "restrict",
            "object": "NetworkAccess",
            "attack_id": "114",
            "attack_name": "Authentication Abuse",
            "description": "An attacker obtains unauthorized access to an application, service or device either through knowledge of the inherent weaknesses",
            "explanation": "The action 'restrict' and object 'NetworkAccess' directly relates with the Authentication Abuse, as the attack seeks to evade these restrictions."
        }
    ]
    
    example_prompt = PromptTemplate(
        input_variables=["class", "action", "object", "attack_id", "attack_name", "description", "explanation"],
        template="""
        - Class: {class}  
        - Action: {action}  
        - Object: {object}  

        **Matched with:**  

        - ID: {attack_id}  
        - Name: {attack_name}  
        - Description: {description}  

        **Explanation:**  
        {explanation}
        """
    )

    few_shot_prompt = FewShotPromptTemplate(
        examples=examples,
        example_prompt=example_prompt,
        prefix="""
        You are an AI assistant specializing in cybersecurity and attack techniques.

    Based on the **retrieved text** and **retrieved CSV attack descriptions**, extract and directly match 
    the **most relevant** attack technique for each class, action and object. Do not forget any class, action, and object.

    **TEXT SOURCE**:
    {docs_content}

    **CSV ATTACK TECHNIQUES**:
    {csv_analysis}

    ---
    **MATCHING TEMPLATE (strictly follow this format)**:
    - **Class Involved**: "<Class>"
    - **Action Performed**: "<Action>"
    - **Object Affected**: "<Object>"
    - **Matched Attack ID**: "<ID from CSV>"
    - **Matched Attack Name**: "<Name from CSV>"
    - **Matched Attack Description**: "<Description from CSV>"

    **Motivation**:
    Explain in **2-3 sentences** why this attack is matched by the action and object.
    ---

    If no exact match is found, state: "No clear connection found.
    Try to find also multiple matches.
    "
        """,
        suffix="",
        input_variables=["docs_content", "csv_analysis"]
    )

    # Using the few-shot prompt template
    formatted_prompt = few_shot_prompt.format(docs_content=docs_content, csv_analysis=csv_analysis)
    
    try:
        response = llm.invoke(formatted_prompt)
        
        # Handle different response formats
        if isinstance(response, str):
            result = response
        elif hasattr(response, 'content'):
            result = response.content
        else:
            result = str(response)
        
        # Cache the result
        #final_answer_cache[cache_key] = result
        
        return {"final_answer": result}
            
    except Exception as e:
        print(f"Error generating final answer: {str(e)}")
        return {"final_answer": f"Error generating final answer: {str(e)}"}

# Build and compile the agent workflow
def create_agent_workflow():
    print("Building agent workflow")
    graph = StateGraph(AgentState)

    # Add nodes
    graph.add_node("retrieve_text", retrieve_text)
    graph.add_node("analyze_csv", analyze_csv)
    graph.add_node("generate_final_answer", generate_final_answer)

    # Define workflow order
    graph.add_edge(START, "retrieve_text")
    graph.add_edge("retrieve_text", "analyze_csv")
    graph.add_edge("analyze_csv", "generate_final_answer")
    graph.add_edge("generate_final_answer", END)

    print("Compiling agent workflow")
    return graph.compile()

# Create function to run the agent
def query_agent(question: str):
    print(f"Querying agent with: {question}")
    try:
        # Create the agent workflow
        agent = create_agent_workflow()
        response = agent.invoke({"question": question})
        print("Agent query completed successfully")
        return response["final_answer"]
    except Exception as e:
        print(f"Error during agent query: {str(e)}")
        return f"Error querying agent: {str(e)}"

# Only run this if the script is executed directly
def RAG_capec_agent():
    print("Retrieving Agent-specific Attack Patterns from CAPEC")
    query = "Retrieve the full content of the provided file"
    print("\nExecuting query:", query)
    result = query_agent(query)
    print("\nMatched Attack Patterns from Capec:", result)
   
    return result 
