from FileExtractor import FileExtractor
from langchain.schema import Document
from langchain_community.vectorstores import FAISS
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain.chains import RetrievalQA

fe = FileExtractor();
# Create OllamaEmbedder
ollama_embedder = OllamaEmbeddings(model="nomic-embed-text")

documents = [
    {"filename": "history.txt", "text": fe.extract_text_from_txt("files/history.txt")},
    {"filename": "science.txt", "text": fe.extract_text_from_txt("files/science.txt")},
    { "filename": "ai_research.txt", "text": fe.extract_text_from_txt("files/ai_research.txt")},
    {"filename": "climate_change.txt", "text": fe.extract_text_from_txt("files/climate_change.txt")},
    {"filename": "cybersecurity.txt", "text": fe.extract_text_from_txt("files/cybersecurity.txt")},
    {"filename": "economics.txt", "text": fe.extract_text_from_txt("files/economics.txt")},
    {"filename": "philosophy.txt", "text": fe.extract_text_from_txt("files/philosophy.txt")},
    {"filename": "space_exploration.txt", "text": fe.extract_text_from_txt("files/space_exploration.txt")},
]

# Generate embeddings
for doc in documents:
    doc["embedding"] = ollama_embedder.embed_query(doc["text"])


# Convert documents into LangChain Document format
doc_objects = [Document(page_content=doc["text"], metadata={"source": doc["filename"]}) for doc in documents]

# Create FAISS vectorstore
vectorstore = FAISS.from_documents(doc_objects, ollama_embedder)

# Load Ollama LLM
llm = ChatOllama(model="llama3.2:latest")

# Set up retriever from FAISS
retriever = vectorstore.as_retriever()

# QA chain
qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)

# Query example
query = "What have provided insights into planetary?"
response = qa_chain.invoke(query)

print("Answer:", response['result'])




# query = "Tell me about technological advancement"
# query_embedding = oe.get_embedding(query)
#
# # Rank documents by similarity
# results = ss.rank_documents(documents, query_embedding)
#
# # Display top result
# print("Most relevant document:", results[0]["filename"])
# print("Content:", results[0]["text"])
