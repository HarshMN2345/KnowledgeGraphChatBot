import streamlit as st
from llama_index.core import VectorStoreIndex, Document, StorageContext, ServiceContext, load_index_from_storage
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core.node_parser import SentenceSplitter
from llama_index.llms.groq import Groq
import warnings
import os
from neo4j import GraphDatabase
import spacy
import chardet
import tempfile
import PyPDF2

warnings.filterwarnings('ignore')

# ---- NEO4J SETUP ----
neo4j_uri = "neo4j+s://db030f0d.databases.neo4j.io"
neo4j_user = "neo4j"
neo4j_password = "TWfNzPqnknMoOAauDueuv4VcUed-CjM2CL-s6H1AKms"
driver = GraphDatabase.driver(neo4j_uri, auth=(neo4j_user, neo4j_password))

# ---- ENVIRONMENT VARIABLES ----
os.environ["GROQ_API_KEY"] = "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"
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# ---- PROMPT TEMPLATE ----
prompt_template = """
Use the following pieces of information to answer the user's question.
If you don't know the answer, just say that you don't know, don't try to make up an answer.

Context: {context}
Graph Insights: {graph_insights}
Question: {question}

Answer the question and provide additional helpful information,
based on the pieces of information and graph insights, if applicable. Be succinct.

Responses should be properly formatted to be easily read.
"""

# Define the context for your prompt
context = "This system contains multiple documents providing examples and solutions for various tasks."

# Load spacy model
@st.cache_resource
def load_spacy_model():
    return spacy.load("en_core_web_sm")

nlp = load_spacy_model()

# Function to extract entities and relationships from documents
def populate_graph(documents, driver, nlp):
    with driver.session() as session:
        for doc in documents:
            doc_text = doc.text
            nlp_doc = nlp(doc_text)
            concepts = [ent.text for ent in nlp_doc.ents if ent.label_ in ["ORG", "PRODUCT"]]

            for concept in concepts:
                session.run("MERGE (:Concept {name: $concept})", concept=concept)

            for i, concept in enumerate(concepts):
                if i + 1 < len(concepts):
                    next_concept = concepts[i + 1]
                    session.run(
                        """
                        MATCH (c1:Concept {name: $concept}), (c2:Concept {name: $next_concept})
                        MERGE (c1)-[:RELATED_TO]->(c2)
                        """,
                        concept=concept, next_concept=next_concept
                    )

# Set up embedding model and LLM
@st.cache_resource
def setup_models():
    embed_model = HuggingFaceEmbedding(model_name="sentence-transformers/all-MiniLM-L6-v2")
    llm = Groq(model="llama3-70b-8192", api_key=GROQ_API_KEY)
    return embed_model, llm

embed_model, llm = setup_models()

# Create service context
service_context = ServiceContext.from_defaults(embed_model=embed_model, llm=llm)

# Function to create or update index
def create_or_update_index(documents):
    text_splitter = SentenceSplitter(chunk_size=1024, chunk_overlap=200)
    nodes = text_splitter.get_nodes_from_documents(documents)
    
    if os.path.exists("./storage_mini"):
        storage_context = StorageContext.from_defaults(persist_dir="./storage_mini")
        index = load_index_from_storage(storage_context, service_context=service_context)
        for doc in documents:
            index.insert(doc)
    else:
        index = VectorStoreIndex.from_documents(documents, service_context=service_context, node_parser=nodes)
    
    index.storage_context.persist(persist_dir="./storage_mini")
    return index

# Query Enhancement with Neo4j
def get_graph_insights(question):
    with driver.session() as session:
        result = session.run(
            """
            MATCH (c:Concept)
            WHERE toLower(c.name) CONTAINS toLower($question)
            OPTIONAL MATCH (c)-[r:RELATED_TO]->(other:Concept)
            RETURN c.name AS concept, collect(other.name) AS related_concepts
            """,
            question=question
        )
        insights = []
        for record in result:
            insights.append(f"Concept: {record['concept']}, Related Concepts: {', '.join(record['related_concepts'])}")
        return "\n".join(insights) if insights else "No relevant graph insights found."

# Streamlit UI
st.title("Document Q&A System")

# File uploader
uploaded_files = st.file_uploader("Upload documents", accept_multiple_files=True, type=['txt', 'pdf', 'docx'])

if uploaded_files:
    documents = []
    for file in uploaded_files:
        try:
            # Create a temporary file
            with tempfile.NamedTemporaryFile(delete=False, mode='wb') as temp_file:
                temp_file.write(file.read())
                temp_file_path = temp_file.name

            # Process the file based on its type
            if file.name.lower().endswith('.pdf'):
                with open(temp_file_path, 'rb') as pdf_file:
                    pdf_reader = PyPDF2.PdfReader(pdf_file)
                    content = ""
                    for page in pdf_reader.pages:
                        content += page.extract_text() + "\n"
            else:
                # For non-PDF files, use the previous method
                with open(temp_file_path, 'rb') as raw_file:
                    raw_content = raw_file.read()
                    detected = chardet.detect(raw_content)
                    file_encoding = detected['encoding']

                with open(temp_file_path, 'r', encoding=file_encoding) as f:
                    content = f.read()
                    
            # Create a Document object
            if content:
                doc = Document(text=content, metadata={"filename": file.name})
                documents.append(doc)

            # Remove the temporary file
            os.unlink(temp_file_path)
        except Exception as e:
            st.error(f"Error processing file {file.name}: {str(e)}")

    # Update the index with new documents
    if documents:
        try:
            index = create_or_update_index(documents)
            populate_graph(documents, driver, nlp)
            st.success(f"Successfully processed {len(documents)} documents.")
        except Exception as e:
            st.error(f"Error updating index or graph: {str(e)}")
    else:
        st.warning("No documents were successfully processed.")

# Query input
question = st.text_input("Enter your question:", "Explain Python?")

if st.button("Ask"):
    if os.path.exists("./storage_mini"):
        try:
            storage_context = StorageContext.from_defaults(persist_dir="./storage_mini")
            index = load_index_from_storage(storage_context, service_context=service_context)
            query_engine = index.as_query_engine(service_context=service_context)
            
            graph_insights = get_graph_insights(question)
            query_prompt = prompt_template.format(context=context, graph_insights=graph_insights, question=question)
            response = query_engine.query(query_prompt)
            st.write(response.response)
        except Exception as e:
            st.error(f"Error querying the index or generating response: {str(e)}")
    else:
        st.warning("No documents have been uploaded yet. Please upload some documents first.")
