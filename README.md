# Valeo Document Assistant

## Description
The Valeo Document Assistant is a Retrieval-Augmented Generation (RAG) system that processes a PDF document, builds a FAISS vector store for efficient similarity search, and generates context-aware answers using a GPT-2 model. Given a user's question, the system retrieves relevant document chunks and formulates an answer based on the provided context.

## Features
- PDF document processing and vector embeddings
- Context-aware question answering
- Docker containerization
- Interactive CLI interface
- GPT-2 medium model for answer generation

## Note 
- Due to limited local resources, GPT-2 is used for answer generation. However, I created a Kaggle notebook that utilizes the Mistral 7B model for better performance. You can access it here: [Kaggle Notebook](https://www.kaggle.com/code/nairmarwa/rag-use-case).

## Project Structure
```
.
├── data/                             # Document storage
├── app.py                            # Main application
├── Dockerfile                        # Container configuration
├── requirements.txt                  # Python dependencies
└── README.md                         # This document
```
## Requirements
To run the program, ensure you have the following dependencies installed:

- Python 3.10+
- Required Python packages (listed in `requirements.txt`):
  - `langchain==0.3.15`
  - `langchain-community==0.3.15`
  - `transformers==4.47.0`
  - `sentence-transformers==3.3.1`
  - `faiss-cpu==1.9.0.post1`
  - `pypdf==5.1.0`
- Docker (if running in a containerized environment)

## Setup Instructions

### 1. Clone the Repository
```sh
git clone <repository-url>
cd <repository-name>
```

### 2. Install Dependencies (For Local Execution)
```sh
pip install --no-cache-dir -r requirements.txt
```

### 3. Build the Docker Image (For Containerized Execution)
```sh
docker build -t valeo-doc-assistant .
```

## Execution

### Running Locally
To run the assistant interactively:
```sh
python app.py
```
To process a single command-line question:
```sh
python app.py "What is the revenue of Valeo in 2022?"
```

### Running with Docker
To start an interactive session:
```sh
docker run -it --rm valeo-doc-assistant
```
To process a single command-line question:
```sh
docker run --rm valeo-doc-assistant "What is the revenue of Valeo in 2022?"
```

## Example Usage

#### Example Input:
```sh
python app.py "What are the key financial highlights of Valeo in 2022?"
```

#### Example Output:
```
Welcome to Valeo Document Assistant!
Type 'exit' to quit.

Enter your question: What's Valeo's approach to autonomous driving?

Answer: We have developed a system for the automatic 
driving of automobiles and trucks that uses artificial 
intelligence and the Blue Ocean Strategy. This system is 
in its early stages in the planning phase, but it will become 
easier to drive as the technology improves. It 
is based on the design of an autonomous driving system that we have developed 
in collaboration with a number of automotive and engineering 
companies. We have developed a basic system that should be able to 
drive without a human driver at any time.

--------------------------------------------------
```

## Technical Notes

- The assistant supports both interactive and command-line execution.
- Initial setup may take some time (model downloads + index creation).
- Uses GPT-2-medium (355M parameters) because of ressource constraits.


## Configuration

To use different documents:
1. Replace `data/Valeo-2022-Universal-Registration-Document.pdf`
2. Delete existing FAISS index:
   ```bash
   rm -rf faiss_index
   ```
3. Rebuild Docker image
