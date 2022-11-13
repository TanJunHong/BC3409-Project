import pandas as pd
import numpy as np
import scipy

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD

from nltk.corpus import brown

from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from gensim.models import FastText

from pipelines.preprocessing.Preprocessor import Preprocessor

import gensim

from tqdm import tqdm

class TextEncoder:
    """
    This section of the pipeline aims to vectorize the texts. Three main methods are implemented here. One applies a more
    classical Bag-Of-Words approach. The next approach utilizes Latent Semantic Analysis to reduce it's dimensionality.
    The last approach utilizes a pre-trained neural network based Embedding layer such as BERT to encode the input.
    """

    def __init__(self):
        self.vectorizer = None
        self.svd = None

    def __apply_tfidf(self, texts_to_cluster: pd.Series, ngrams_wanted: tuple) -> (TfidfVectorizer, scipy.sparse):
        """
        Internal function to apply tf-idf vectorization on ngrams.
        First parses the texts into ngrams specified and applies tf-idf on the resultant ngrams.

        :param texts_to_cluster: Pandas Series of texts to be vectorized.
        :param ngrams_wanted: Range of ngrams to be specified in the vectorized output. (1, 3) implies 1-gram to 3-gram.
        :return: Vectorizer and vectorized texts
        """

        vectorizer = self.vectorizer if self.vectorizer is not None else TfidfVectorizer(stop_words="english",
                                                                                         ngram_range=ngrams_wanted)
        word_vec = vectorizer.fit_transform(texts_to_cluster)

        self.vectorizer = vectorizer
        return vectorizer, word_vec

    def __apply_svd(self, matrix_to_cluster: scipy.sparse, n_components: int = 300) -> (TruncatedSVD, np.ndarray):
        """
        Internal function to apply SVD dimensionality reduction on the output from apply_tfidf.
        This is used injunction with apply_tfidf to run Latent Semantic Analysis on our text inputs.

        https://stackoverflow.com/questions/48424084/number-of-components-trucated-svd

        :param matrix_to_cluster: Tfidf Matrix
        :param n_components: Dimensionality targeted
        :return: SVD Decomposer and the reduced vector
        """
        if matrix_to_cluster.shape[1] <= n_components:
            return matrix_to_cluster
        svd = TruncatedSVD(n_components=n_components) if self.svd is None else self.svd
        reduced_vec = svd.fit_transform(matrix_to_cluster)
        self.svd = svd
        return svd, reduced_vec

    def apply_lsa(self, texts_to_cluster: pd.Series, ngrams_wanted: tuple) \
            -> (TfidfVectorizer, scipy.sparse, TruncatedSVD, scipy.sparse):
        """
        Implements Latent Semantic Analysis by running Truncated SVD on TF-IDF outputs. This is done to reduce the
        dimensionality of texts using a BOW approach.

        Follows the reference: https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.TruncatedSVD.html


        :param texts_to_cluster:
        :param ngrams_wanted:
        :return: A tuple of vectorizers and matrices trained and transformed is output here.
        """
        vectorizer, word_vec = self.__apply_tfidf(texts_to_cluster, ngrams_wanted)
        svd, reduced_vec = self.__apply_svd(word_vec)

        return vectorizer, word_vec, svd, reduced_vec

    def apply_bert(self):
        pass

    def apply_doc2vec(self, texts_to_cluster: pd.Series, ndim: int = 300, print_progress: bool = False) -> (Doc2Vec, np.ndarray):
        processed_corpus = [TaggedDocument(tokens, [idx]) for idx, tokens in enumerate(texts_to_cluster)]
        doc_vec_model = Doc2Vec(vector_size=ndim, min_count=2, epochs=40)
        doc_vec_model.build_vocab(processed_corpus)

        if print_progress:
            processed_corpus = tqdm(processed_corpus)

        text_vector = np.array([doc_vec_model.infer_vector(doc.words) for doc in processed_corpus])

        return doc_vec_model, text_vector

    def apply_fasttext(self, texts_to_cluster: pd.Series, ndim: int = 300) -> (FastText, np.ndarray):
        def extract_embedding(model, x):
            try:
                return fast_text_model.wv[" ".join(x)]
            except:
                return np.zeros(ndim)

        corpus = texts_to_cluster.copy()

        corpus[corpus.map(len) == 0] = ["people"]
        corpus = corpus.map(tuple)


        fast_text_model = FastText(size=ndim, window=3, min_count=2)
        fast_text_model.build_vocab(sentences=corpus)
        fast_text_model.train(corpus, total_examples=len(corpus), epochs=10)

        fast_text_vector = corpus.map(lambda x: extract_embedding(fast_text_model, x))

        fast_text_vector = np.array(fast_text_vector.to_list())

        return fast_text_model, fast_text_vector

if __name__ == '__main__':
    # sentences = brown.sents(categories=['news', 'editorial', 'reviews'])
    # unprocessed = pd.Series([" ".join(x) for x in sentences])

    reducer = TextEncoder()

    # _, _, _, processed = reducer.apply_lsa(unprocessed, ngrams_wanted=(1, 3))

    # np.save("../../temp_data/brown_lsa_data.npy", processed, allow_pickle=False)

    data = pd.read_csv("../../../../data/train2.csv")

    subtexts = data["text"][:1000]

    ner_chunks = Preprocessor.named_entity_chunking(subtexts)
    vectorizer, vec = reducer.apply_doc2vec(ner_chunks)
