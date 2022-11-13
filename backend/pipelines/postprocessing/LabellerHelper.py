import pandas as pd
import re


def find_top_terms_for_each_cluster(list_of_dfs: list, num_top: int) -> [pd.DataFrame]:
    """
    Sort all dataframes by the frequency in descending order and take the top n rows.
    :param list_of_dfs: List of dataframes with one column of frequency
    :param num_top: N number to take from the top.
    :return: List of top terms for each cluster.
    """

    sorted_data_map = map(lambda x: x.sort_values("freq", ascending=False), list_of_dfs)
    return [df.head(num_top) for df in sorted_data_map]


def find_common_top_terms(list_of_top_term_dfs: list) -> pd.Series:
    """
    Gets a series of all terms, then gets the value counts to get freq of all terms across all data frames.
    :param list_of_top_term_dfs: List of data frames.
    :return: Returns a series of all terms and their frequency.
    """

    all_top_terms = pd.Series([top_term for top_term_df in list_of_top_term_dfs for top_term in top_term_df["term"]])
    return all_top_terms.value_counts()


def filter_common_top_terms(top_terms_series: pd.Series, num_wanted: int,
                            list_of_cluster_top_terms_dfs: [pd.DataFrame]) -> [pd.DataFrame]:
    """

    :param top_terms_series: A pandas Series where each row has a index of the term they are representing
    :param num_wanted: Top n number to filter off
    :param list_of_cluster_top_terms_dfs: List of cluster top terms dfs to filter for top terms
    :return: Filtered list of cluster top terms df
    """
    too_common_terms = top_terms_series.index.values[:num_wanted]
    return [df.loc[~df['term'].isin(too_common_terms)] for df in list_of_cluster_top_terms_dfs]


def describe_lda_topic(topic: str):
    """

    :param topic:
    :return:
    """
    topic_components = topic.split(" + ")
    topic_components_details = [re.sub("\"", "", topic_component).split("*") for topic_component in topic_components]

    topic_descriptors_weights, topic_descriptors = zip(*topic_components_details)
    return topic_descriptors_weights, topic_descriptors


def describe_lda_topics(lda_model, is_hdp: bool = False):
    """
    Retrieves topics from a gensim lda or hdp model and generates the descriptor for each of them.
    :param lda_model:
    :param is_hdp:
    :return:
    """
    topics_list = lda_model.show_topics() if is_hdp else lda_model.show_topics(-1)
    lda_topics = [topic_element[1] for topic_element in topics_list]
    topics_descriptors = {idx: describe_lda_topic(topic)[1] for idx, topic in enumerate(lda_topics)}
    return topics_descriptors
