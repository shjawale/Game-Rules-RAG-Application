# Board Game RAG Application
A **Retrieval-Augmented Generation (RAG)** system built to answer questions about board games. It uses **ChromaDB** for vector storage and **Google Gemini 2.0 Flash** for generation and evaluation. **Google Search Grounding** is integrated to verify facts against live web data.

## Getting Started
### 1. Create a Virtual Environment

Python 3.12 is recommended to avoid ConfigError issues because the project uses Pydantic v1 via the ChromaDB client.

### Create the environment
```
python3.12 -m venv .venv
```

### Activate the environment
```
source .venv/bin/activate        #On macOS/Linux:
.\venv\Scripts\activate          #On Windows
```

### 2. Set Up Google API Key

A Google Gemini API key is needed to handle embeddings, text generation, and search grounding.

2.1.  Get an API key from the Google AI Studio.

2.2.  Create a file named .env in the project root directory. See the .env.example file.

2.3.  Add the key to the file:

``` 
GOOGLE_API_KEY=your_actual_api_key_here
```

### 3. Install Dependencies

Install the required libraries from requirements.txt using pip.
```
google-genai
chromadb
pandas
seaborn
```

Run the installation:
```
pip install -r requirements.txt
```

## Features
* **Automated Data Retrieval**: Downloads a board game dataset via ```kagglehub``` for immediate use.
* **Semantic Vector Search**: Uses ```GeminiEmbeddingFunction``` to index and retrieve game descriptions.
* **Search Grounding**: Integrates Google Search to provide citations and verify verify answer accuracy.
* **AI-as-a-Judge Evaluation**: Includes an "AI-as-a-Judge" section that scores responses on a scale of 1–5 based on groundedness and completeness.
* **Similarity Visualization**: Generates a heatmap to compare semantic similarity between RAG-based and Search-grounded results.


## What is RAG?
Retrieval-Augmented Generation (RAG) is the core architecture of this application. It prevents AI "hallucinations" by giving the model a specific set of rules to reference before it generates a response.

The process uses three main stages:

*    **Retrieval: When a question about a game is asked, for example, "How do I play Twilight Imperium?", the system searches the ChromaDB vector database to to isolate the rows that contain the most relevant data for that query.
*    **Augmentation**: The system merges the user’s question with the retrieved data into a structured instruction, labeling each database entry as an individual "PASSAGE" to help the model clearly distinguish between different games or records.
*    **Generation**: The Gemini model processes these specific passages to generate an answer. By grounding the response only in the provided database rows, it avoids relying on general training data and ensures the output is accurate to the selected games.

### Why use it?
Standard AI models may confuse games with similar names or miss specific version updates. This RAG setup ensures that if a game is in the dataset, the answer will be accurate to that specific text.


## Quality Assurance & Verification
Accuracy and reliability are maintained through automated checks and external data verification.

### Automated Evaluation
An **AI-as-a-Judge** framework measures response quality based on strict criteria:

*    **Metrics**: Responses are graded on Groundedness (sticking to the provided text), Instruction Following, Completeness, and Fluency.
*    **Scoring**: The system uses a specialized prompt (QA_PROMPT) to generate a step-by-step critique, assigning a score from 1 (Wrong) to 5 (Excellent).
*    **Structured Output**: By using a Python Enum, the evaluator forces the AI to provide a machine-readable score that can be used for automated testing or performance tracking.


### Google Search Grounding
Google Search Grounding ensures factual accuracy and provides verifiable sources.

*    **Real-Time Verification**: The grounding tool connects to the live web to double-check dates, creators, or specific rules that may have been updated since the dataset was published.
*    **Citations**: Footnotes and a bibliography are generated automatically. Each claim in the AI's response is mapped to a specific grounding_chunk with a title and source URL.
*    **Trust Signals**: Every claim is cross-referenced against grounding_metadata to confirm supporting evidence exists for each statement. If supporting evidence for a claim is missing, the script can be configured to retry or flag the information as unverified.


## Technical Insights
Implementing this RAG pipeline on a sparse dataset of game rules provided several practical insights into how grounding and evaluation metrics work together to ensure accuracy.

*    **Managing Data Sparsity**: Since the dataset had very little information per game, I used similarity scores to monitor retrieval quality. I found that when scores were low, the Groundedness metric became a critical safety check, ensuring the model prioritized accuracy over Completeness and refused to "invent" rules when the local data was missing.
*    **Search Grounding Reliability**: As the primary embedding database was thin, adding external search grounding became the backbone of the system’s reliability. It allowed the pipeline to maintain high scores by pulling in accurate rules from the web whenever the local retrieval fell short.
*    **Constraint Adherence**: Through the evaluator and Python Enum scoring, I confirmed that the model followed negative constraints (e.g., "Do not make up rules"). This is critical for rule-based datasets, where an incorrect rule is worse than no answer at all.


## Project Workflow
The implementation follows a modular pipeline. You can follow along in the code via the internal comments:

*    Environment Setup: Library imports and API configuration.
*    Database Creation: Embedding the game dataset for retrieval.
*    RAG Pipeline: Finding relevant documents and generating answers.
*    Evaluation Suite: Scoring the results via the "AI-as-a-Judge" framework.
*    Grounding Tools: Running external search and calculating similarity scores.

## Acknowledgments
This project was developed as a Capstone for the Google 5-Day Gen AI Intensive Course. While following Google's recommended reference architectures for RAG and evaluation, I independently implemented the pipeline to see how LLM grounding performs on sparse, rule-based datasets.
