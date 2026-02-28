#!/usr/bin/env python
# coding: utf-8


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import os
from dotenv import load_dotenv


from google import genai
from google.genai import types
import chromadb
import functools


CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
HOME_DIR = os.getenv("HOME")


# Load .env with explicit path
env_path = os.path.join(CURRENT_DIR, ".env")
load_dotenv(dotenv_path=env_path, override=True)


# Set up API key
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")


# Create a RAG application to help you understand the rules of any board game
#  Explore available models
client = genai.Client(api_key=GOOGLE_API_KEY)

for m in client.models.list():
    if "embedContent" in m.supported_actions:
        print(m.name)


#  Add data
import kagglehub
dataset_path = kagglehub.dataset_download("sujaykapadnis/board-games") #/versions/1/board_games.csv")
print("Path to dataset files:", dataset_path)

DATASET_PATH = os.path.join(dataset_path, "board_games.csv")


df = pd.read_csv(DATASET_PATH)

print(df["image"][21:23])
df.head()


#  Convert data df to list of documents with each document as the name of game + descriptions
#start with creating a single document.
n = len(df)
maxlen = 100
start = 0
documents = [str(df["game_id"][i]) + " " + str(df["name"][i]) + ": " + str(df["description"][i]) for i in range(start, start+maxlen)]


# Creating the embedding database with ChromaDB

from chromadb import Documents, EmbeddingFunction, Embeddings
from google.api_core import retry
from google.genai import types

# Define a helper to retry when per-minute quota is reached.
is_retriable = lambda e: (isinstance(e, genai.errors.APIError) and e.code in {429, 503})

class GeminiEmbeddingFunction(EmbeddingFunction):
    # Specify whether to generate embeddings for documents, or queries
    document_mode = True

    @retry.Retry(predicate=is_retriable)
    def __call__(self, input: Documents) -> Embeddings:
        if self.document_mode:
            embedding_task = "retrieval_document"
        else:
            embedding_task = "retrieval_query"

        response = client.models.embed_content(
            model="models/text-embedding-004",
            contents=input,
            config=types.EmbedContentConfig(
                task_type=embedding_task,
            ),
        )
        return [e.values for e in response.embeddings]


#  create a Chroma database client

DB_NAME = "boardgamedatabase"

embed_fn = GeminiEmbeddingFunction()
embed_fn.document_mode = True

chroma_client = chromadb.Client()
db = chroma_client.get_or_create_collection(name=DB_NAME, embedding_function=embed_fn)

db.add(documents=documents, ids=[str(i) for i in range(len(documents))])


# Retrieval: Find relevant documents¶

# Switch to query mode when generating embeddings.
embed_fn.document_mode = False

# Search the Chroma DB using the specified query.
query = "What's the first step in Twilight Imperium?\nHow do you play Twilight Imperium?\nHow do you win the game?\nGive an example of a move in Twilight Imperium.\nWhat's the most interesting part of the game?\nWhat is a common mistake players make?"

result = db.query(query_texts=[query], n_results=2)
[all_passages] = result["documents"]

print(all_passages[0])


# Augmented generation: Answer the question

query_oneline = query.replace("\n", " ")

prompt = f"""You will answer questions using the reference passage included below. Make sure to respond in complete sentences, be comprehensive and concise, and include all relevant background information. Ensure you break down complicated concepts and strike a friendly and conversational tone. If the passage is irrelevant to the answer, you may ignore it.

QUESTION: {query_oneline}
"""

# Add the retrieved documents to the prompt.
for passage in all_passages:
    passage_oneline = passage.replace("\n", " ")
    prompt += f"PASSAGE: {passage_oneline}\n"

print(prompt)


AG_all_qs_answer = client.models.generate_content(
    model="gemini-2.0-flash",
    contents=prompt
)

print(AG_all_qs_answer.text)


# Evaluation and structured output

# Define an evaluator

'''terse_guidance = "Answer the following question in a single sentence, or as close to that as possible."
moderate_guidance = "Provide a brief answer to the following question, use a citation if necessary, but only enough to answer the question."
cited_guidance = "Provide a thorough, detailed answer to the following question, citing the document and supplying additional background information as much as possible."
guidance_options = {
    'Terse': terse_guidance,
    'Moderate': moderate_guidance,
    'Cited': cited_guidance,
}'''

questions = [
    "What's the first step in Twilight Imperium?",
    "How do you play Twilight Imperium?",
    "How do you win the game?"
    "Give an example of a move in Twilight Imperium."
    "What's the most interesting part of the game?"
    "What is a common mistake players make?"
]

if not questions:
  raise NotImplementedError('Add some questions to evaluate!')


@functools.cache
def answer_question(questions: tuple, guidance: str = '') -> str:
  """Generate an answer to the question using the uploaded document and guidance."""
  config = types.GenerateContentConfig(
      temperature=2.0,
      #system_instruction=guidance,
  )
  response = client.models.generate_content(
      model='gemini-2.0-flash',
      config=config,
      contents=[questions, documents],
  )

  return response.text


import enum

QA_PROMPT = """\
You are an expert evaluator. Your task is to evaluate the quality of the responses generated by AI models.
We will provide you with the user prompt which might include more than one question and the corresponding AI-generated responses.
You should read the user prompt carefully to analyze the task, and then evaluate the quality of each of the responses separately based on the rules provided in the Evaluation section below.

# Evaluation
## Metric Definition
You will be assessing the AI model's question answering quality, which measures the overall quality of the answer to a question in the user prompt. Pay special attention to length constraints, such as in X words or in Y sentences. The instruction for performing a question-answering task is provided in the user prompt. The response should not contain information that is not present in the context (if it is provided).

You will assign the writing response a score from values 1 to 5, following the Rating Rubric and Evaluation Steps.
Give step-by-step explanations for your scoring, and only choose scores from values 1 to 5.

## Criteria Definition
Instruction following: The response demonstrates a clear understanding of the question answering task instructions, satisfying all of the instruction's requirements.
Groundedness: The response contains information included only in the context if the context is present in the user prompt. The response does not reference any outside information.
Completeness: The response completely answers the question with sufficient detail.
Fluent: The response is well-organized and easy to read.

## Rating Rubric
5: (Excellent). The answer follows instructions, is grounded, complete, and fluent.
4: (Good). The answer follows instructions, is grounded, complete, but is not very fluent.
3: (Ok). The answer mostly follows instructions, is grounded, answers the question partially and is not very fluent.
2: (Bad). The answer does not follow the instructions very well, is incomplete or not fully grounded.
1: (Wrong). The answer does not follow the instructions, is wrong and not grounded.

## Evaluation Steps
STEP 1: Assess the response in aspects of instruction following, groundedness,completeness, and fluency according to the criteria for each question separately. Be comprehensive and use complete sentences.
STEP 2: Give a separate score for each question based on the rubric.

# User Inputs and AI-generated Response
## User Inputs
### Prompt
{prompt}

## AI-generated Response
{response}
"""

class AnswerRating(enum.Enum):
  EXCELLENT = '5'
  GOOD = '4'
  OK = '3'
  BAD = '2'
  WRONG = '1'


@functools.cache
def eval_answer(prompt, ai_response, n=1):
  """Evaluate the generated answer against the prompt/question used."""
  chat = client.chats.create(model='gemini-2.0-flash')

  # Generate the full text response.
  response = chat.send_message(
      message=QA_PROMPT.format(prompt=[prompt, documents], response=ai_response)
  )
  verbose_eval = response.text

  # Coerce into the desired structure.
  structured_output_config = types.GenerateContentConfig(
      response_mime_type="text/x.enum",
      response_schema=AnswerRating,
  )
  print('\n\n')
  
  response = chat.send_message(
      message="Convert the final score.",
      config=structured_output_config,
  )
  structured_eval = response.parsed
  
  return verbose_eval, structured_eval

print(AG_all_qs_answer.text)
text_eval, struct_eval = eval_answer(prompt=tuple(questions), ai_response=tuple(AG_all_qs_answer.text))
display(print(text_eval))
print()
print(struct_eval)


# Google Search Grounding

## Use search grounding

# Ask for information without search grounding.
response = client.models.generate_content(
    model='gemini-2.0-flash',
    contents=questions
)

print(response.text)


# And now re-run the same query with search grounding enabled.
config_with_search = types.GenerateContentConfig(
    tools=[types.Tool(google_search=types.GoogleSearch())],
)

def query_with_grounding():
    grounding_response = client.models.generate_content(
        model='gemini-2.0-flash',
        contents=questions,
        config=config_with_search,
    )
    return grounding_response.candidates[0]


rc = query_with_grounding()
print(rc.content.parts[0].text)


while not rc.grounding_metadata.grounding_supports or not rc.grounding_metadata.grounding_chunks:
    # If incomplete grounding data was returned, retry.
    rc = query_with_grounding()

chunks = rc.grounding_metadata.grounding_chunks
for chunk in chunks:
    print(f'{chunk.web.title}: {chunk.web.uri}')


print(rc.grounding_metadata.search_entry_point.rendered_content)


from pprint import pprint

supports = rc.grounding_metadata.grounding_supports
for support in supports:
    print(support.to_json_dict())


import io

markdown_buffer = io.StringIO()

# Print the text with footnote markers.
markdown_buffer.write("Supported text:\n\n")
for support in supports:
    markdown_buffer.write(" * ")
    markdown_buffer.write(
        rc.content.parts[0].text[support.segment.start_index : support.segment.end_index]
    )

    for i in support.grounding_chunk_indices:
        chunk = chunks[i].web
        markdown_buffer.write(f"<sup>[{i+1}]</sup>")

    markdown_buffer.write("\n\n")


markdown_buffer.write("Citations:\n\n")
for i, chunk in enumerate(chunks, start=1):
    markdown_buffer.write(f"{i}. [{chunk.web.title}]({chunk.web.uri})\n")

print(markdown_buffer.getvalue())


# Embeddings and similarity scores

# Search with tools

def show_response(response):
    for p in response.candidates[0].content.parts:
        if p.text:
            display(print(p.text))
        elif p.inline_data:
            display(Image(p.inline_data.data))
        else:
            print(p.to_json_dict())
    
        display(print('----'))


config_with_search = types.GenerateContentConfig(
    tools=[types.Tool(google_search=types.GoogleSearch())],
    temperature=0.0,
)

chat = client.chats.create(model='gemini-2.0-flash-001')

response = chat.send_message(
    message="When was Twilight Imperium created?",
    config=config_with_search,
)

show_response(response)


print(response.text)


config_with_code = types.GenerateContentConfig(
    tools=[types.Tool(code_execution=types.ToolCodeExecution())],
    temperature=0.0,
)

response = chat.send_message(
    message="Now plot this as a seaborn chart.",
    config=config_with_code,
)

show_response(response)


## Calculate similarity scores

texts = [AG_all_qs_answer.text,
         rc.content.parts[0].text]

print(AG_all_qs_answer.text)
print('\n\n')
print(rc.content.parts[0].text)

response = client.models.embed_content(
    model='models/text-embedding-004',
    contents=texts,
    config=types.EmbedContentConfig(task_type='semantic_similarity'))


def truncate(t: str, limit: int = 40) -> str:
  """Truncate labels to fit on the chart."""
  if len(t) > limit:
    return t[:limit-3] + '...'
  else:
    return t

truncated_texts = [truncate(t) for t in texts]


# Set up the embeddings in a dataframe.
df = pd.DataFrame([e.values for e in response.embeddings], index=truncated_texts)
# Perform the similarity calculation
sim = df @ df.T
# Draw!
sns.heatmap(sim, vmin=0, vmax=1, cmap="Blues");


print()
print(sim)

