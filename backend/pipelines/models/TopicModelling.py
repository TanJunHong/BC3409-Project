import gensim
from gensim.corpora import Dictionary
from gensim.models.ldamodel import LdaModel
from gensim.models.hdpmodel import HdpModel

import smart_open


smart_open.open = smart_open.smart_open

"""
Provides a wrapper to initialize, fit and predict classical Topic Modelling models based on the input data using the
models in gensim for Topic Modelling.
"""


def perform_lda(dictionary: Dictionary, bow_corpus: list, num_topics: int = 20, single_core: bool = False) -> LdaModel:
    """
    Wrapper to consolidate generating an LdaModel using a corpus, dictionary and number of topics.
    Uses Latent Dirichlet Allocation to generate our topics.
    :param dictionary: Dictionary used to generate the bow corpus.
    :param bow_corpus: Bow generated from a corpus of documents.
    :param num_topics: Number of topics expected to model.
    :param single_core: If facing issues, set single_core to True to use LdaModel instead of LdaMulticore
    :return: A LdaModel trained on the number of topics, corpus and dictionary fed.
    """

    # https://towardsdatascience.com/evaluate-topic-model-in-python-latent-dirichlet-allocation-lda-7d57484bb5d0
    if single_core:
        return gensim.models.LdaModel(bow_corpus, num_topics=num_topics, id2word=dictionary, passes=10)

    return gensim.models.LdaMulticore(bow_corpus, num_topics=num_topics, id2word=dictionary, passes=10, workers=None)


def perform_hdp(dictionary: Dictionary, bow_corpus: list) -> HdpModel:
    """
    Wrapper to generate a HdpModel using a corpus and dictionary.
    Uses Hierarchical Dirichlet Process to generate our topics.
    :param dictionary: Dictionary used to generate the bow corpus.
    :param bow_corpus: Bow generate from a corpus of documents.
    :return: A HdpModel trained on the corpus and dictionary fed.
    """
    return gensim.models.HdpModel(bow_corpus, dictionary)
