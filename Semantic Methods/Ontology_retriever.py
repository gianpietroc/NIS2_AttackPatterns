import rdflib
import re
from langchain_ollama import ChatOllama
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain  
from langchain_ollama import OllamaEmbeddings
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import TextLoader

# Load your ontology file (change path if needed)
ontology_path = "ontology.owl"
g = rdflib.Graph()
g.parse(ontology_path) 

def extract_local_name(uri):
    if isinstance(uri, str):
        return re.sub(r".*[#/]", "", uri)  # Removes everything before # or /
    return uri

# Convert RDF triples to readable text without URIs

# Function to search ontology using SPARQL with substring matching
def search_by_substring(substring):
    sparql_query = f"""
PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
PREFIX owl: <http://www.w3.org/2002/07/owl#>
PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
PREFIX onto: <http://www.semanticweb.org/gianpietro/ontologies/2022/6/untitled-ontology-39#>

SELECT DISTINCT ?class_n ?property ?value
WHERE {{
  {{
    # Direct properties
    ?class_n ?property ?value .
  }}
  UNION
  {{
    # Handling equivalent class intersections
    ?class_n owl:equivalentClass ?equiv .
    ?equiv owl:intersectionOf|owl:unionOf ?list .
    ?list rdf:rest*/rdf:first ?value .
    BIND("intersectionMember" AS ?property)
  }}
  UNION
  {{
    # Handling Restrictions (someValuesFrom)
    ?restriction rdf:type owl:Restriction ;
                 owl:onProperty ?property ;
                 owl:someValuesFrom ?value .
    ?class_n owl:equivalentClass|rdfs:subClassOf ?restriction .
  }}
  UNION
  {{
    # Handling nested restrictions (deeper levels)
    ?class_n owl:equivalentClass|rdfs:subClassOf ?nested .
    ?nested owl:intersectionOf|owl:unionOf ?list1 .
    ?list1 rdf:rest*/rdf:first ?item1 .
    
    {{
      # Level 1 Restrictions
      ?item1 rdf:type owl:Restriction ;
             owl:onProperty ?property ;
             owl:someValuesFrom ?value .
      
             OPTIONAL {{
        ?value owl:intersectionOf ?intersection .
        ?intersection rdf:rest*/rdf:first ?item2 .
        ?item2 rdf:type owl:Restriction ;
              owl:onProperty ?nestedProperty ;
              owl:someValuesFrom ?nestedValue .
    }}
    }}
    
  }}
  
  # Filter out structural properties
  FILTER(?property != rdf:type)
  FILTER(?property != owl:intersectionOf)
  FILTER(?property != rdf:rest)
  FILTER(?property != rdf:first)
  FILTER (
        CONTAINS(LCASE(STR(?class_n)), LCASE("{substring}")))
}}
ORDER BY ?class_n ?property ?value
"""
    qres = g.query(sparql_query)
    pattern = r'^N[a-zA-Z0-9]*$'

    # Format results
    results = []
    for row in qres:
        class_n = row.class_n if row.class_n else "None"
        property = row.property if row.property else "None"
        value = row.value if row.value else "None"
        property = extract_local_name(property)
        if re.match(pattern, value):
            continue
        if property == "comment":
            continue 
        if  property != "None" and value != "None" and property != "subClassOf" and property != "equivalentClass" \
        and property != "intersectionMember":
            results.append(f"Class: {extract_local_name(class_n)}, Action: {property}, \
                            Object: {extract_local_name(value)}")
    
    return results if results else ["No matches found."]

def rag_llm(keyword, results):
    with open("RAG_docs/results.txt", "w") as file:
        file.write(results)

    llm = ChatOllama(model="llama3.2:1b", model_provider="ollama")
    embeddings = OllamaEmbeddings(model="llama3.2:1b")

    # Use FAISS instead of in-memory storage
    loader = TextLoader("RAG_DOCS/results.txt")
    docs = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1024, chunk_overlap=100)
    all_splits = text_splitter.split_documents(docs)
    
    vector_store = FAISS.from_documents(all_splits, embeddings)
    retriever = vector_store.as_retriever(search_kwargs={"k": 5})

    # Define structured prompt with explicit extraction instructions
    PROMPT_TEMPLATE = """
    You are an AI assistant specializing in legal document analysis. 
    Extract relevant legal actions and responsibilities from the retrieved text. 
    Your output should be structured in the format: 

    **Class**: <Class Name>  
    **Action**: <Action>  
    **Object**: <Object>  

    If multiple matches exist, list them all. Only use the retrieved documents for context.  
    Do NOT generate information that is not explicitly stated.

    ------
    **Retrieved Context:**
    {context}

    **Question:** {question}

    **Answer:**
    """

    prompt = PromptTemplate(template=PROMPT_TEMPLATE, input_variables=["context", "question"])

    # Retrieve relevant documents
    def retrieve(question):
        retrieved_docs = retriever.get_relevant_documents(question)
        return retrieved_docs

    # Generate answer using retrieved context
    def generate_answer(question, retrieved_docs):
        docs_content = "\n\n".join(doc.page_content for doc in retrieved_docs)
        formatted_prompt = prompt.format(context=docs_content, question=question)
        response = llm.invoke(formatted_prompt)
        return response.content

    # Run the RAG pipeline
    question = f"What are the actions that {keyword} must take?"
    retrieved_docs = retrieve(question)

    # Debug: Print retrieved docs
    print("Retrieved Documents:")
    for i, doc in enumerate(retrieved_docs):
        print(f"Document {i+1}: {doc.page_content}\n")

    answer = generate_answer(question, retrieved_docs)
    print("Generated Answer:\n", answer)

    return answer

def ontology_agent(user_query):
    # Check if the user wants to search for entities
    if "search" in user_query.lower():
        keyword = user_query.split("search for ")[-1].strip()
        print("Retrieving Agent-related Measures from the ontology")
        results = search_by_substring(keyword)
        results =  "\n".join(results)
        
        with open("RAG_docs/results.txt", "w") as file:
            file.write(results)
        #rag_llm(keyword, results)


import os
def Retrieving_agent_ontology():
    if os.path.exists("RAG_docs/results_tmp.txt"):
        os.remove("RAG_docs/results_tmp.txt")
    user_input = input("Ask the ontology agent: ")
    ontology_agent(user_input)
