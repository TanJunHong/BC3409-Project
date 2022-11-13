import pandas as pd

from nltk.corpus import brown

from pipelines.preprocessing.Preprocessor import Preprocessor
from pipelines.preprocessing.TextEncoder import TextEncoder
from pipelines.models import TopicModelling
from pipelines.models.ClusterManager import ClusterManager
from pipelines.postprocessing.Labeller import Labeller

import json


class PipelineManager:
    """
    Pipelines for different analysis are provided here. Stuff like vectorizers and cluster models are not cached as they
    do not work well on interpreting a second, new set of data except for KMeans.

    """
    def __init__(self):
        # self.preprocessor = Preprocessor()
        self.encoder = TextEncoder()
        self.cluster_manager = ClusterManager()
        # self.labeller = Labeller()

    def run_optimal_pipeline(self, texts: pd.Series) -> (pd.Series, [pd.Series]):
        chunk_type = "NER"
        cluster_type = "kmeans"

        return self.perform_chunking_clustering_labelling(texts, chunk_type, cluster_type)

    def perform_chunking_clustering_labelling(self, texts: pd.Series, chunk_type="NER", cluster_type="kmeans",
                                              num_top_terms: int = 30, num_too_common_top_terms: int = 25) \
            -> (pd.Series, [pd.Series]):

        if chunk_type == "NER":
            chunk_texts = Preprocessor.named_entity_chunking(texts)
        else:
            chunk_texts = Preprocessor.key_terms_chunking(texts)
        vectorizer, text_vector = self.encoder.apply_doc2vec(chunk_texts)

        cluster_labels, _ = self.cluster_manager.predict_algorithms[cluster_type](reduced_vec=text_vector)
        cluster_label_col = "labels"
        text_col = "text"

        df = pd.DataFrame(zip(cluster_labels, chunk_texts), columns=[cluster_label_col, text_col])

        descriptors = Labeller.cluster_naming(df, cluster_col=cluster_label_col, num_top_terms=num_top_terms,
                                              num_too_common_top_terms=num_too_common_top_terms, text_present=True,
                                              text_col=text_col)

        return df, descriptors

    def perform_clustering_labelling(self, texts: pd.Series, cluster_type: str = "kmeans", ngram_range: (int, int) = (1, 1),
                                     num_top_terms: int = 30, num_too_common_top_terms: int = 25) \
            -> (pd.Series, [pd.Series]):
        # Pre-processing steps
        # BERT vs LSA
        # BERT vs NLTK word_tokenize
        vectorizer, tfidf_matrix, _, lsa_matrix = self.encoder.apply_lsa(texts, ngram_range)

        # Topic Modelling Steps
        cluster_labels, _ = self.cluster_manager.predict_algorithms[cluster_type](reduced_vec=lsa_matrix)
        cluster_label_col = "labels"

        df = pd.DataFrame(cluster_labels, columns=[cluster_label_col])

        # Labelling steps
        descriptors = Labeller.cluster_naming(df, cluster_col=cluster_label_col, vectorizer=vectorizer,
                                              matrix=tfidf_matrix, num_top_terms=num_top_terms,
                                              num_too_common_top_terms=num_too_common_terms)

        return cluster_labels, descriptors

    def perform_hdp(self, texts, filter_extreme_above=0.1, keep_n=100000):
        return self.__run_gensim_topic_modelling(texts, num_topics=None, filter_extreme_above=filter_extreme_above,
                                                 keep_n=keep_n)

    def perform_lda(self, texts, num_topics=20, filter_extreme_above=0.1, keep_n=100000):
        return self.__run_gensim_topic_modelling(texts, num_topics, filter_extreme_above, keep_n)

    @staticmethod
    def __run_gensim_topic_modelling(texts, num_topics=None, filter_extreme_above=0.1, keep_n=100000):
        # Pre-processing Steps
        dictionary, bow_corpus = Preprocessor.tokenize_dict_bow(texts, filter_extreme_above, keep_n)

        # Topic Modelling Steps
        if num_topics is None:
            model = TopicModelling.perform_hdp(dictionary, bow_corpus)
        else:
            model = TopicModelling.perform_lda(dictionary, bow_corpus, num_topics)

        # Labelling Steps
        descriptors, document_labels = Labeller.label_gensim_topics(model, bow_corpus)

        return document_labels, descriptors, model


if __name__ == "__main__":
    sentences = brown.sents(categories=['news', 'editorial', 'reviews'])
    unprocessed = pd.Series([" ".join(x) for x in sentences])

    pipeline = PipelineManager()
    labels, descriptor = pipeline.perform_clustering_labelling(unprocessed, "kmeans", (2, 4))

    input = {
        "sentences": unprocessed.tolist()
    }

    output = {
        "labels": labels.tolist(),
        "label_descriptors": descriptor
    }

    data_output = pd.DataFrame(list(zip(unprocessed.tolist(), labels.tolist())), columns=["sentences", "labels"])

    data_output.to_csv("./sample_output_to_input.csv")

    with open("./sample_input.json", "w") as f:
        json.dump(input, f)

    with open("./sample_output.json", "w") as f:
        json.dump(output, f)

    print(labels)
    print()
    print(descriptor)
