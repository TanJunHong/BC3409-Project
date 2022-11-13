import nltk
import numpy as np
import pandas as pd

from gensim.corpora import Dictionary

import string

from nltk import SnowballStemmer, WordNetLemmatizer
from nltk.corpus import wordnet, brown

import spacy
import textacy

from tqdm import tqdm

class Preprocessor:
    """
    This section of the pipeline focuses on pre-processing the input texts. Two key functionalities are provided.

    1. Pre-processing of texts by stemming or lemmatization, punctuation and stopwords removal.
    2. Creation of Dictionaries and bag of words given the texts.

    Tokenization is done using the word_tokenizer in NLTK.
    Stemming is done through a Snowball Stemmer and Lemmatization is done using a WordNet Lemmatizer.
    """

    @staticmethod
    def __tokenize(text_df: pd.Series) -> pd.Series:
        """
        Internal function to tokenize a pandas Series of texts. Swap tokenizer to change for the pipelines here.

        :param text_df: Pandas Series with the texts to be tokenized.
        :return: Pandas Series where each row is a tokenized text.
        """
        # return text_df.str.split(" ").values
        return text_df.map(lambda x: nltk.word_tokenize(x))

    @staticmethod
    def __generate_dictionary_bow(tokenized_texts: pd.Series, filter_extreme_above: float = 0.1,
                                  keep_n: int = 100000) -> (Dictionary, list):
        """
        Internal function to generate a bag of words from tokenized texts.

        :param tokenized_texts: Pretokenized texts in a pandas Series.
        :param filter_extreme_above: Texts to filter from the dictionary if they are above this threshold.
        :param keep_n: Size to restrict the dictionary to.
        :return: Returns a dictionary to vectorize future data as well as a list of bagged words.
        """

        # Define defaults here
        filter_extreme_above = 0.1 if filter_extreme_above is None else filter_extreme_above
        keep_n = 100000 if keep_n is None else keep_n

        dictionary = Dictionary(tokenized_texts)
        # Filter words that are too frequent and probably fillers
        dictionary.filter_extremes(no_above=filter_extreme_above, keep_n=keep_n)
        bow_corpus = [dictionary.doc2bow(doc) for doc in tokenized_texts]
        return dictionary, bow_corpus

    @staticmethod
    def tokenize_dict_bow(text_series: pd.Series, filter_extreme_above: float = 0.1, keep_n: int = 100000):
        """
        Provides functionality to create a bag of words given a series of texts. Preprocessing can be chosen or ignored
        as the two steps are separate.

        :param text_series: Corpus of the documents the dictionary is to be made from. Each element in the series is a
        document or could be a sentence.
        :param filter_extreme_above: Threshold of the dictionary for the word to be too common and should be filtered.
        :param keep_n: Size of dictionary
        :return: Dictionary for future encoding and bag of words for current input.
        """
        tokens = Preprocessor.__tokenize(text_series)
        return Preprocessor.__generate_dictionary_bow(tokens, filter_extreme_above, keep_n)


    @staticmethod
    def __punc_remover(input_texts: pd.Series) -> pd.Series:
        """
        Internal function to remove punctuations for pre-processing.

        :param input_texts: Non-tokenized texts to remove punctuation.
        :return: Non-tokenized texts with punctuations removed.
        """

        return input_texts.map(
            lambda x: x.translate(str.maketrans("", "", string.punctuation))
        )

    @staticmethod
    def __get_wordnet_pos(tag):
        """
        References: https://stackoverflow.com/questions/15586721/wordnet-lemmatization-and-pos-tagging-in-python
        return WORDNET POS compliance to WORDENT lemmatization (a,n,r,v)
        """
        if tag.startswith('J'):
            return wordnet.ADJ
        elif tag.startswith('V'):
            return wordnet.VERB
        elif tag.startswith('N'):
            return wordnet.NOUN
        elif tag.startswith('R'):
            return wordnet.ADV
        else:
            # As default pos in lemmatization is Noun
            return wordnet.NOUN

    @staticmethod
    def input_stem_lemmatize(input_texts: pd.Series, process_type: str = "lemmatize", tokenize=True) -> pd.Series:
        """
        Preprocessing pipeline to remove punctuations, stopwords and apply stemming or lemmatization. Both processes
        uses the Natural Language Tool Kit. Stemming utilizes the SnowballStemmer and lemmatization uses the WordNet
        Lemmatizer.

        Raises Exception when an unsupported process is given.

        :param input_texts: Texts to be preprocessed. Applied on a series.
        :param process_type: "stem" or "lemmatize" are supported.
        :return: A pandas series containing the cleaned text. Output is not tokenized.
        """
        ignore_punctuation = True
        ignore_stopwords = True

        if ignore_punctuation:
            input_texts = Preprocessor.__punc_remover(input_texts)

        terms_series = Preprocessor.__tokenize(input_texts)

        if process_type == "stem":
            stemmer = SnowballStemmer("english", ignore_stopwords=ignore_stopwords)
            words = terms_series.map(lambda terms: [stemmer.stem(word) for word in terms])

        elif process_type == "lemmatize":
            lemmatizer = WordNetLemmatizer()

            tags = terms_series.map(lambda x: nltk.pos_tag(x))
            tags_list = tags.tolist()
            words = pd.Series([
                [lemmatizer.lemmatize(word, pos=Preprocessor.__get_wordnet_pos(tag)) for word, tag in sentence]
                for sentence in tags_list
            ])

        elif process_type == "none":
            words = terms_series

        else:
            raise Exception("Unsupported type")

        if not tokenize:
            return words.map(lambda x: " ".join(x))
        else:
            return words

    @staticmethod
    def named_entity_chunking(texts: pd.Series, print_progress: bool = False) -> pd.Series:
        nlp = spacy.load("en_core_web_sm", enable=["ner"], disable=[])

        unwanted_entities = ["TIME", "DATE", "CARDINAL"]

        iter_gen = tqdm(nlp.pipe(texts, n_process=1), total=len(texts)) if print_progress else nlp.pipe(texts)

        entities_tokens = [
            [token.text for token in doc.ents if token.label_ not in unwanted_entities]
            for doc in iter_gen
        ]

        return pd.Series(entities_tokens)

    @staticmethod
    def key_terms_chunking(texts: pd.Series, print_progress: bool = False) -> pd.Series:
        # Potentially remove static to rely on caching nlp to potentially speed up the loadings.
        nlp = spacy.load("en_core_web_sm")

        iter_gen = tqdm(nlp.pipe(texts, n_process=-1), total=len(texts)) if print_progress else nlp.pipe(texts)

        nlp_parsed = [textacy.extract.keyterms.textrank(text_doc) for text_doc in iter_gen]
        return pd.Series(nlp_parsed)

if __name__ == '__main__':
    sentences = brown.sents(categories=['news', 'editorial', 'reviews'])
    unprocessed = pd.Series([" ".join(x) for x in sentences])

    processed_stem = Preprocessor.input_stem_lemmatize(unprocessed, "stem")
    processed_lemm = Preprocessor.input_stem_lemmatize(unprocessed, "lemmatize")

    print(processed_stem)
    print(processed_lemm)

    dictionary, bow = Preprocessor.tokenize_dict_bow(unprocessed)

    print(bow)
    print(dictionary)

