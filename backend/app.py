import base64
import logging
import os
import time
import urllib.parse
from io import BytesIO
from json import load, loads
from tempfile import SpooledTemporaryFile

import numpy
import pandas
import requests
from flask import Flask, jsonify, request
from flask.wrappers import Response
from flask_cors import CORS
from newspaper import Article
from pandas import DataFrame, read_csv
from pymongo import MongoClient
from pymongo.server_api import ServerApi
from pymongo.collection import Collection
from pymongo.database import Database
from werkzeug.datastructures import ImmutableMultiDict

import summariser
from pipelines.PipelineManager import PipelineManager

app: Flask = Flask(__name__)
CORS(app)
username = urllib.parse.quote_plus(os.environ["MONGODB_USERNAME"])
password = urllib.parse.quote_plus(os.environ["MONGODB_PASSWORD"])
host = urllib.parse.quote_plus(os.environ["MONGODB_HOST"])
client: MongoClient = MongoClient(
    host="mongodb+srv://%s:%s@%s/?retryWrites=true&w=majority" % (username, password, host), server_api=ServerApi("1"))
logging.basicConfig(level=logging.INFO)


def mongo_import(file_temp_path: SpooledTemporaryFile, file_name: str, db_name: str, col_name: str) -> int:
    """
    Imports a file to a mongo collection, given the file path, file name, database name and collection name.
    @param col_name:
    @param file_temp_path: The path to the temp file
    @param file_name: The name of the file
    @param db_name: The name of the database
    @param col_name: The name of the collection
    @return: The number of documents imported
    """
    db: Database = client[db_name]
    col: Collection = db[col_name]

    if file_name.endswith(".csv"):
        data: DataFrame = read_csv(filepath_or_buffer=file_temp_path)
        payload: list = loads(s=data.to_json(orient="records"))
    elif file_name.endswith(".json"):
        payload: list = [load(fp=file_temp_path)]
    else:
        raise NotImplementedError(f"File type not supported: {file_temp_path}")

    col.delete_many(filter={})

    col.insert_many(documents=payload)
    return col.count_documents(filter={})


@app.route(rule="/load_dataset_names", methods=["POST"])
def load_dataset_names() -> Response:
    """
    Loads dataset names.
    @return: Dataset names
    """

    db: Database = client["bc3409_summary"]
    cols: list = db.list_collection_names()

    return jsonify({"dataset_names": cols})


@app.route(rule="/load_labels", methods=["POST"])
def load_labels() -> Response:
    """
    Loads the labels from mongo database, given dataset name.
    @return: Labels formatted for display as word cloud
    """
    dataset: str = request.form["dataset"]

    db: Database = client["bc3409_input"]
    col: Collection = db[dataset]

    occurrence_count: dict = {}
    for document in col.find():
        label: str = document["label"]
        if label in occurrence_count:
            occurrence_count[label] += 1
        else:
            occurrence_count[label] = 1

    word_cloud: list = [{"text": text, "value": value} for text, value in occurrence_count.items()]

    return jsonify({"word_cloud": word_cloud})


@app.route(rule="/load_labels_list", methods=["POST", "GET"])
def load_labels_list() -> Response:
    """
    Loads the labels from mongo database, given dataset name.
    @return: Labels in a list
    """
    dataset: str = "default"  # request.form["dataset"]

    db: Database = client["bc3409_input"]
    col: Collection = db[dataset]

    return jsonify({"labels": col.distinct(key="label")})


@app.route(rule="/load_some_articles", methods=["GET"])
def load_some_articles() -> Response:
    """
    Loads the articles from mongo database, given dataset name, label, skip_count and limit_count.
    @return: 2 articles
    """
    dataset: str = request.args.get("dataset")
    label: str = request.args.get("label")
    limit_count: int = 2

    db: Database = client["bc3409_input"]
    col: Collection = db[dataset]

    articles: list = []

    for doc in col.find({"label": label}).limit(limit=limit_count):
        if "image" not in doc:
            archive_url: str = doc["archive"].replace("id_/http", "/http")
            article: Article = Article(url=archive_url)
            article.download()
            article.parse()
            if article.top_image != "":
                img = BytesIO(initial_bytes=requests.get(url=article.top_image).content).getvalue()
                base64_img: str = base64.b64encode(s=img).decode(encoding="utf-8")
                col.update_one(filter={"_id": doc["_id"]}, update={"$set": {"image": base64_img}})
                doc["image"] = base64_img
            else:
                col.update_one(filter={"_id": doc["_id"]}, update={"$set": {"image": "None"}})

        if "image" in doc and doc["image"] == "None":
            del doc["image"]

        doc["_id"] = str(doc["_id"])
        articles.append(doc)

    db: Database = client["bc3409_summary"]
    col: Collection = db[dataset]
    sentences_list: list = col.find_one({"label": label})["sentences"]

    return jsonify({"articles": articles, "summary": "\n".join(sentences_list)})

@app.route(rule="/load_articles", methods=["POST"])
def load_articles() -> Response:
    """
    Loads the articles from mongo database, given dataset name, label, skip_count and limit_count.
    @return: Articles with _id, title and image
    """
    dataset: str = request.form["dataset"]
    label: str = request.form["label"]
    skip_count: int = int(request.form["skip"])
    limit_count: int = int(request.form["limit"])

    db: Database = client["bc3409_input"]
    col: Collection = db[dataset]

    articles: list = []

    logging.info(f"Loading article images with the follow params: "
                 f"dataset: {dataset}, label: {label}, skip: {skip_count}, limit: {limit_count}")

    skip_count = min(max(col.count_documents(filter={"label": label}) - limit_count, 0), skip_count)

    for doc in col.find({"label": label}).skip(skip=skip_count).limit(limit=limit_count):
        if "image" not in doc:
            archive_url: str = doc["archive"].replace("id_/http", "/http")
            article: Article = Article(url=archive_url)
            article.download()
            article.parse()
            if article.top_image != "":
                img = BytesIO(initial_bytes=requests.get(url=article.top_image).content).getvalue()
                base64_img: str = base64.b64encode(s=img).decode(encoding="utf-8")
                col.update_one(filter={"_id": doc["_id"]}, update={"$set": {"image": base64_img}})
                doc["image"] = base64_img
            else:
                col.update_one(filter={"_id": doc["_id"]}, update={"$set": {"image": "None"}})

        if "image" in doc and doc["image"] == "None":
            del doc["image"]

        doc["_id"] = str(doc["_id"])
        articles.append(doc)

    db: Database = client["bc3409_summary"]
    col: Collection = db[dataset]
    sentences_list: list = col.find_one({"label": label})["sentences"]

    return jsonify({"articles": articles, "summary": sentences_list})


@app.route(rule="/upload", methods=["POST"])
def upload() -> Response:
    """
    Handles user file upload, and calls backend.
    @return: Response if successful
    """
    db_name: str = "bc3409_input"

    data: ImmutableMultiDict = request.files
    dataset: str = request.form["dataset"]

    col_count: int = mongo_import(file_temp_path=data["File"].stream, file_name=data["File"].filename,
                                  db_name=db_name, col_name=dataset)

    logging.info(f"Imported {col_count} documents")

    db: Database = client[db_name]
    col: Collection = db[dataset]
    texts: pandas.Series = pandas.Series([document["text"] for document in col.find()])

    pipeline_manager: PipelineManager = PipelineManager()

    logging.info(f"Running clustering pipeline")
    cluster_labels: numpy.ndarray
    descriptors: list[str]
    cluster_labels, descriptors = pipeline_manager.perform_clustering_labelling(texts=texts)
    df: pandas.DataFrame = pandas.DataFrame(data=list(zip(texts.tolist(), cluster_labels.tolist())),
                                            columns=["sentences", "labels"])

    descriptor_labels: list = [descriptors[cluster_labels[i].item()] for i in range(len(cluster_labels))]

    i: int = 0
    for doc in col.find():
        col.update_one(filter={"_id": doc["_id"]}, update={"$set": {"label": descriptor_labels[i]}})
        i += 1

    logging.info(f"Running summarization pipeline")

    start: float = time.time()
    results: list[list[str]] = summariser.run_summariser(df=df)
    end: float = time.time()
    logging.info(f"Summarization took {end - start} seconds")

    df = pandas.DataFrame(data=list(zip(results, descriptors)),
                          columns=["sentences", "label"])

    db_name = "bc3409_summary"
    db = client[db_name]
    col = db[dataset]
    col.delete_many(filter={})
    col.insert_many(documents=df.to_dict(orient="records"))

    return jsonify({"status": "success"})


if __name__ == "__main__":
    app.run()
