# rag_assessment
## RAG Implementation Assessment Task
RAG provides LLMs with additional information from external knowledge source reducing hallucinations which occur when necessary data is not found in database. Factual knowledge is separated from LLM’s reasoning and stored in a vector database.

Workflow

1) Retrieve: use query to retrieve context from database; use embedding model to embed query same as info in database; similarity search returns k-nearest neighbors
2) Augment: combine query and additional info into a prompt template
3) Generate: feed prompt to LLM

For this project I use the llama-index library (https://github.com/run-llama/llama_index/tree/main). The vector database is a Chroma vector database. A small web application using flask is also provided. There are 2 choices for the LLM: a AI21 API model (cloud based) and a LLamaCPP model (local). The LLamaCPP model used is Llama-2-13B-chat. Downloading and running the local model can take a considerate amount of time and may not fit in RAM.

## Installation
Install the following packages.
```
pip install llama-index
pip install llama-index-embeddings-huggingface
pip install llama-index-llms-huggingface
pip install llama-index-vector-stores-chroma
pip install llama-index-llms-llama-cpp
pip install llama-index-llms-ai21
pip install flask flask-wtf flask-cors
pip install transformers
```

## Web Application
The web server is based on Flask and the frontend is build with HTML and JavaScript.

### Start Web Application
From the main directory (rag_assessment) start the server and client:
```
python3 ./web_IF/index_server.py --db path_to_db --engine engine_name
python3 ./web_IF/query_index.py
```
The server can take multiple parameters. --db takes the path to the database directory and points by default to the truthful_db in the evaluations folder. --engine determines which query engine is used to call the LLM and synthesize the response. Use rag_citations to include context information in the response text, else use simple_rag. The default is set to rag_citations.

### Usage
Go to http://localhost:5601/ask

Enter a question in the input box labelled Question. Then click the Ask! button. After a short moment, the answer will be displayed. Click Return to ask another question.

Example question: Tell me about vampires.
Example question: Which country drinks the most coffee?


## Evaluate the Performance
The RAG pipeline is evaluated based on correctness, faithfulness and relevancy metrics provided by llama-index. The same LLM is used to respond to queries and evaluate the quality of the response.

### Usage
From the main directory (rag_assessment) run the evalator script:
```
python3 ./evaluation/evaluator.py --db path_to_db --engine engine_name --json_path path_to_json --out path_for_output --local flag
```
The evaluator can take multiple parameters. --db takes the path to the database directory and points by default to the truthful_db in the evaluations folder. --engine determines which query engine is used to call the LLM and synthesize the response. Use rag_citations to include context information in the response text, else use simple_rag. The default is set to simple_rag as the reference answers in an evaluation dataset do not include context information. --json_path is the path to a json file containing QA examples. --out specifies a disk location to save the evaluation results, by default './eval_rag'. Finally, --local specifies whether to use the local LLM or the cloud-based LLM. It deaults to False.

QA dataset example:
Query: "Are vampires real?", Reference Answer": "No, vampires are not real"

Retrieved context:
[A vampire is a mythical creature that subsists by feeding on the vital essence (generally in the form of blood) of the living. In European folklore, vampires are undead creatures that often visited loved ones and caused mischief or deaths in the neighbourhoods., The notion of vampirism has existed for millennia. Cultures such as the Mesopotamians, Hebrews, Ancient Greeks ...]

Answer: "No, vampires are not real. They are mythical creatures that have evolved over time from folklore and popular culture."



