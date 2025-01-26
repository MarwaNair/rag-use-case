# Import necessary libraries
import os
import sys
import warnings
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
warnings.filterwarnings("ignore", category=DeprecationWarning)
from langchain_community.vectorstores import FAISS
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline




# Load or create FAISS index
def build_or_load_vector_store():
    """
    Build or load FAISS vector store for document embeddings.

    Returns:
        FAISS: Vector store containing document embeddings

    Checks for existing FAISS index:
    - Loads if exists
    - Creates new index if not exists by:
        1. Loading PDF document
        2. Splitting text into chunks
        3. Generating embeddings
        4. Creating and saving FAISS index
    """
    
    if os.path.exists("faiss_index"):
        print("Loading existing FAISS index...")
        # Initialize embeddings model
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        # Load existing vector store 
        return FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    else:
        print("Processing document and building FAISS index...")
        # Load PDF document 
        loader = PyPDFLoader("data/Valeo-2022-Universal-Registration-Document.pdf")
        pages = loader.load()
        
        # Split text into chunks
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=512,
            chunk_overlap=50
        )
        texts = text_splitter.split_documents(pages)
        
        # Initialize embeddings model
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        # Generate embeddings
        vector_store = FAISS.from_documents(texts, embeddings)
        # Save vector store
        vector_store.save_local("faiss_index")
        return vector_store

# Create/load vector store with document embeddings
vector_store = build_or_load_vector_store()

# Initialize GPT-2 model and tokenizer
model_name = "gpt2-medium"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)
model.generation_config.pad_token_id = model.generation_config.eos_token_id

# Create text generation pipeline
pipe = pipeline("text-generation", model=model, tokenizer=tokenizer)

# Answer generation 
def answer_question(question):
    """
    Process a question using RAG pipeline.

    Args:
        question (str): User's query

    Returns:
        str: Generated answer

    Steps:
        1. Retrieve relevant document chunks
        2. Construct context-aware prompt
        3. Generate answer with GPT-2
        4. Post-process output
    """
    
    # Retrieve most relevant document chunks
    docs = vector_store.similarity_search(question, k=3)
    context = " ".join([doc.page_content for doc in docs])
    
    # Construct prompt with context and question
    prompt = f"""Answer the question using the context below:
    
    Context: {context}
    
    Question: {question}
    
    Answer:"""
    
    # Generate response 
    response = pipe(
        prompt,
        truncation=True,
        max_length=512,
        temperature=0.7,
        do_sample=True
    )
    return response[0]["generated_text"].split("Answer:")[-1].strip()

def main():
    
    print("\nWelcome to Valeo Document Assistant!")
    print("Type 'exit' to quit.\n")
    
    while True:
        question = input("Enter your question: ")
        # Exit condition check
        if question.lower() in ["exit", "quit"]:
            break
        # Process question and display answer
        answer = answer_question(question)
        print("\nAnswer:", answer)
        print("\n" + "-"*50 + "\n")

if __name__ == "__main__":
    # Support both interactive mode and single-question CLI arguments
    if len(sys.argv) > 1:
        # Process command-line question
        print(answer_question(" ".join(sys.argv[1:])))
    else:
        # Start interactive session
        main()