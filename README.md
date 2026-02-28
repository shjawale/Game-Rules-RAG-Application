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
* **Dataset**: Automatically downloads a board game dataset via ```kagglehub```.
* **Vector Search**: Uses ```GeminiEmbeddingFunction``` to index and retrieve game descriptions.
* **Grounding**: Integrates Google Search to provide citations and verify the accuracy of answers.
* **Evaluation**: Includes an "AI-as-a-Judge" section that scores responses on a scale of 1–5 based on groundedness and completeness.
* **Visualization**: Generates a heatmap to compare semantic similarity between RAG-based and Search-grounded answers.


## What is RAG?
Retrieval-Augmented Generation (RAG) is the core architecture of this application. It solves the problem of AI "hallucinations" by giving the model a specific set of facts to look at before it speaks.

The process uses three main steps:

**Retrieval**: A question, such as "How do I play Twilight Imperium?", is sent to the system. The system searches the ChromaDB vector database to find specific game descriptions and rules.
    
**Augmentation**: The system adds the retrieved game rules to the original question. It creates a detailed instruction: "Using these specific rules [Rules], answer this question [Question]."
    
**Generation**: The Gemini model uses the provided rules to generate a response. The answer is based only on the data provided, not the model's general training data.

#### Why use it here?

Standard AI models may confuse board games with similar names or miss specific version updates. This RAG setup ensures that if the game is in the CSV file, the answer will be accurate to that specific entry.


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
