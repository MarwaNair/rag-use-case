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
git clone https://github.com/MarwaNair/rag-use-case.git
cd rag-use-case
```

### 2. Install Dependencies (For Local Execution)
```sh
pip install --no-cache-dir -r requirements.txt
```
OR

### 2. Build the Docker Image (For Containerized Execution)
```sh
docker build -t valeo-doc-assistant .
```

## Execution

### Running Locally
```sh
python app.py
```


### Running with Docker

```sh
docker run --rm -it valeo-doc-assistant
```

## Example Usage

#### Example Input:
```sh
python app.py
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
