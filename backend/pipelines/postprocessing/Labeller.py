import scipy

import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer

from nltk.corpus import brown

from gensim.models.hdpmodel import HdpModel
from gensim.models.ldamodel import LdaModel
from gensim.models.ldamulticore import LdaMulticore
from gensim.interfaces import TransformationABC

from pipelines.preprocessing.Preprocessor import Preprocessor
from pipelines.preprocessing.TextEncoder import TextEncoder
from pipelines.models.TopicModelling import perform_lda, perform_hdp
from pipelines.models.ClusterManager import ClusterManager
from pipelines.postprocessing import LabellerHelper


class Labeller:
    """
    Provides static methods to call to label a group of documents. Two main methods of labelling are through topic
    modelling algorithms and another is using common ngrams to label clusters. Topic Modelling algorithms used are
    Latent Dirichlet Allocation and Hierarchical Dirichlet Process, both of which describes topics with a list of words.
    The other uses tfidf and ngrams to label clusters by their most frequent ngrams.
    """

    @staticmethod
    def __get_labels_gensim_from_bow(gensim_model, bow_vectors: list) -> pd.Series:
        """
        Predicts the topic for each document in the bow_vectors using the gensim model provided. Suitable for Latent
        Dirichlet Allocation or Hierarchical Dirichlet Process. Returns the index of topic for each document.
        :param gensim_model:
        :param bow_vectors:
        :return: Returns a series of topics for this model.
        """
        topics = []
        outputs = gensim_model[bow_vectors]
        for doc in outputs:
            if len(doc) == 0:
                topics.append(-1)
            else:
                topics.append(sorted(doc, key=lambda tup: -1 * tup[1])[0][0])

        return pd.Series(topics)

    @staticmethod
    def label_gensim_topics(gensim_model: TransformationABC, bow_vectors) -> ({int: [str]}, pd.Series):
        """
        Labels the texts in the bow based on their topics and generates labels for each topic. Returns a dictionary
        describing each topic and the label for each document as a pandas Series.
        :param gensim_model: Accepts a LdaModel, LdaMulticore or HdpModel that has already been trained.
        :param bow_vectors: A bag of words that has been tokenized with a dictionary.
        :return: Returns a dictionary where the key is the topic index and the value is a list of tokens that describe
        the topic. It also returns a pandas series of topic index each row belongs to.
        """

        is_hdp = type(gensim_model) == HdpModel
        is_lda = type(gensim_model) == LdaModel or type(gensim_model) == LdaMulticore

        assert is_hdp or is_lda

        gensim_labels = Labeller.__get_labels_gensim_from_bow(gensim_model, bow_vectors)
        topic_descriptors = LabellerHelper.describe_lda_topics(gensim_model, is_hdp)

        return topic_descriptors, gensim_labels

    @staticmethod
    def __describe_cluster(cluster_element_indices: list, vectorizer: TfidfVectorizer,
                           tfidf_matrix: scipy.sparse, text_present: bool = True,
                           text_series: pd.Series = None) -> pd.DataFrame:
        """
        Extracts the count values for all documents in the cluster provided and returns the frequency for each word in a
        dataframe.
        :param cluster_element_indices:
        :param vectorizer:
        :param tfidf_matrix:
        :param text_present:
        :param text_series:
        :return: A pandas DataFrame with two columns. Term and how frequent that term apears in this cluster.
        """

        if text_present:
            vectorizer = CountVectorizer(tokenizer=lambda doc: doc, lowercase=False)
            counts = vectorizer.fit_transform(text_series).sum(axis=0)
            words_freq = [(word, counts[0, idx]) for word, idx in vectorizer.vocabulary_.items()]

        else:
            subset_vectorized_texts = tfidf_matrix[cluster_element_indices]
            counts = subset_vectorized_texts.sum(axis=0)
            words_freq = [(word, counts[0, idx]) for word, idx in vectorizer.vocabulary_.items()]

        return pd.DataFrame(words_freq, columns=['term', 'freq'])

    @staticmethod
    def __describe_clusters(overall_df: pd.DataFrame, cluster_col: str, tfidf_vectorizer: TfidfVectorizer,
                            tfidf_matrix: scipy.sparse, text_present: bool = True,
                            text_col: str = "text") -> [pd.DataFrame]:
        """
        Groups the data frame of texts by their cluster label and describes each cluster by the frequency of words in
        each cluster. Requires a pre-trained vectorizer and tfidf matrix of this dataset.
        :param overall_df: Requires minimally a single column of the cluster label.
        :param cluster_col: Name of column that is used to indicate the cluster of each row.
        :param tfidf_vectorizer: Used to get the vocabulary back from the indices.
        :param tfidf_matrix: Used to get the count frequency back from the indices.
        :param text_present:
        :param text_col:
        :return: A list of dataframes where each dataframe describes the cluster by showing the frequency of each ngram.
        """
        list_of_cluster_dfs = overall_df.groupby(cluster_col)
        return [
            Labeller.__describe_cluster(cluster_df.index.values, tfidf_vectorizer, tfidf_matrix, text_present,
                                        cluster_df[text_col])
            for group, cluster_df in list_of_cluster_dfs
        ]

    @staticmethod
    def cluster_naming(dataframe: pd.DataFrame, cluster_col: str, vectorizer: TfidfVectorizer = None,
                       matrix: scipy.sparse = None, num_top_terms: int = 20, num_too_common_top_terms: int = 7,
                       text_present: bool = False, text_col: str = "text") -> [pd.Series]:
        """
        Describes clusters in the dataframe. Provides a layer to generate n grams from a pretrained vectorizer and tfidf
        matrix. N-grams used in descriptor is affected by the vectorizer trained. Too common terms are filtered off to
        reduce commonality of terms.

        :param dataframe: Minimally a dataframe with a column of the predicted cluster label of each row.
        :param cluster_col: Label of the column representing the cluster labels
        :param vectorizer: TfidfVectorizer that is trained on the dataset
        :param matrix: Tfidf Matrix representing the dataset
        :param num_top_terms: Number of top terms to retrieve
        :param num_too_common_top_terms: Number of top terms to keep.
        :param text_present:
        :parma text_col:
        :return: A list of pandas Series describing each cluster. The index of the list is the cluster group it is
        describing.
        """
        cluster_trigrams_dfs = Labeller.__describe_clusters(dataframe, cluster_col, vectorizer, matrix,
                                                            text_present, text_col)
        cluster_top_terms = LabellerHelper.find_top_terms_for_each_cluster(cluster_trigrams_dfs, num_top_terms)
        common_top_terms = LabellerHelper.find_common_top_terms(cluster_top_terms)

        filtered_cluster_top_terms = LabellerHelper.filter_common_top_terms(common_top_terms,
                                                                            num_too_common_top_terms,
                                                                            cluster_top_terms)
        descriptive_terms = [i['term'] for i in filtered_cluster_top_terms]

        return descriptive_terms


if __name__ == '__main__':

    sentences = brown.sents(categories=['news', 'editorial', 'reviews'])
    unprocessed = pd.Series([" ".join(x) for x in sentences])

    # Provide tests for running labeller with cluster and tfidf
    encoder = TextEncoder()
    vectorizer, matrix, _, lsa_matrix = encoder.apply_lsa(unprocessed, (3, 3))

    cluster_manager = ClusterManager()
    kmeans_labels, _ = cluster_manager.predict_kmeans(lsa_matrix, 10)

    df = pd.DataFrame([unprocessed, pd.Series(kmeans_labels)]).T
    df.columns = ["texts", "labels"]

    cluster_names = Labeller.cluster_naming(df, "labels", vectorizer, matrix)

    # Provide tests for running labeller with lda and hdp
    dictionary, bow = Preprocessor.tokenize_dict_bow(unprocessed)
    model = perform_lda(dictionary, bow)
    model2 = perform_hdp(dictionary, bow)

    results = Labeller.label_gensim_topics(model, bow)
    results2 = Labeller.label_gensim_topics(model2, bow)

    print(results)
    print()
    print(results2)
