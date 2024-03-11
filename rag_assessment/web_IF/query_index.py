from flask import Flask
from flask import request
from flask import render_template
from flask_wtf import FlaskForm
from wtforms import StringField, TimeField, DateField, SubmitField, SelectField
from wtforms.validators import DataRequired, Length, Email, EqualTo
from flask import render_template, redirect, url_for, request
from datetime import datetime

import os
import chromadb

from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core import (
    SimpleDirectoryReader,
    VectorStoreIndex,
    StorageContext,
)
from llama_custom.retriever import VectorDBRetriever
from multiprocessing.managers import BaseManager

from llama_custom.rag_application import get_embed_model, get_rag_string_engine, load_llm_model_cloud

app = Flask(__name__)

manager = BaseManager(("", 5602), b"password")
manager.register("query_index")
manager.connect()


@app.route("/")
def home():
    greeting = "Hello there, Ace"
    return render_template('index.html', greet=greeting)


@app.route("/ask")
def ask_q():
    return render_template('base.html', title='Chat Interface')


@app.route("/query", methods=["GET"])
def query_index():
    """
    Gets the query response for a text included in URL and prints it! -> somehow can't print the context
    http://localhost:5601/query?text=what did the author do growing up
    :return:
    """
    global index
    query_text = request.args.get("text", None)
    if query_text is None:
        return (
            "No text found, please include a ?text=blah parameter in the URL",
            400,
        )
    #query_engine = index.as_query_engine()
    response = manager.query_index(query_text)._getvalue()
    # str(response), 200
    return render_template('answer.html', query=query_text, response=str(response))


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5601)
