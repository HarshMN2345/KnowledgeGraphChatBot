import streamlit as st
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, StorageContext, ServiceContext, load_index_from_storage
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core.node_parser import SentenceSplitter
from llama_index.llms.groq import Groq
import warnings
import os
from neo4j import GraphDatabase
import spacy

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
context = "This directory contains multiple documents providing examples and solutions for various programming tasks."

# Data ingestion: load all files from a directory
@st.cache_resource
def load_documents():
    directory_path = r"C:\Users\harsh\OneDrive\Documents\offerletterdocs"
    reader = SimpleDirectoryReader(input_dir=directory_path)
    return reader.load_data()

documents = load_documents()

# Load spacy model (you can choose a different model)
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

# Populate the Neo4j graph
populate_graph(documents, driver, nlp)

# Split the documents into nodes
@st.cache_resource
def get_nodes():
    text_splitter = SentenceSplitter(chunk_size=1024, chunk_overlap=200)
    return text_splitter.get_nodes_from_documents(documents)

nodes = get_nodes()

# Set up embedding model and LLM
@st.cache_resource
def setup_models():
    embed_model = HuggingFaceEmbedding(model_name="sentence-transformers/all-MiniLM-L6-v2")
    llm = Groq(model="llama3-70b-8192", api_key=GROQ_API_KEY)
    return embed_model, llm

embed_model, llm = setup_models()

# Create service context
service_context = ServiceContext.from_defaults(embed_model=embed_model, llm=llm)

# Create vector store index
@st.cache_resource
def create_index():
    vector_index = VectorStoreIndex.from_documents(documents, service_context=service_context, node_parser=nodes)
    vector_index.storage_context.persist(persist_dir="./storage_mini")
    return vector_index

vector_index = create_index()

# Load the index from storage
@st.cache_resource
def load_index():
    storage_context = StorageContext.from_defaults(persist_dir="./storage_mini")
    return load_index_from_storage(storage_context, service_context=service_context)

index = load_index()

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

# Query Engine Setup
query_engine = index.as_query_engine(service_context=service_context)

# Streamlit UI
st.title("Document Q&A System")

question = st.text_input("Enter your question:", "Explain Python?")

if st.button("Ask"):
    graph_insights = get_graph_insights(question)
    query_prompt = prompt_template.format(context=context, graph_insights=graph_insights, question=question)
    response = query_engine.query(query_prompt)
    st.write(response.response)