from typing import List

import numpy as np
from gensim.models.word2vec import Word2Vec


def vectorizer(
    corpus: List[List[str]], model: Word2Vec, num_features: int = 100
) -> np.ndarray:
    """
    This function takes a list of tokenized text documents (corpus) and a pre-trained
    Word2Vec model as input, and returns a matrix where each row represents the
    vectorized form of a document.

    Args:
        corpus : list
            A list of text documents that needs to be vectorized.

        model : Word2Vec
            A pre-trained Word2Vec model that will be used to vectorize the corpus.

        num_features : int
            The size of the vector representation of each word. Default is 100.

    Returns:
        corpus_vectors : numpy.ndarray
            A 2D numpy array where each row represents the vectorized form of a
            document in the corpus.
    """
    corpus_vectors = []  # Initialize empty list to store the vectors for all documents

    # Iterate over all documents in corpus
    for doc in corpus:
        # Initialize empty array to store the vector representation of the document
        word_vectors = np.zeros((len(doc), num_features))
        # Initialize counter to keep track of the number of words in the document
        word_count = 0
        # Iterate over all words in the document
        for idx, word in enumerate(doc):
            # Check if the word is present in the vocabulary
            if word in model.wv:
                # If yes, then add its vector to the word_vectors array
                word_vectors[idx] = model.wv[word]
                # Increment the counter by 1
                word_count += 1

        # Compute the average by dividing the sum of all word vectors by the number of words
        if word_count == 0:
            avg_vector = np.zeros(
                (num_features,)
            )  # If no word is present in the vocabulary, then return a vector of zeros
        else:
            avg_vector = np.mean(
                word_vectors, axis=0
            )  # Else, return the average vector of the document

        # Append the average vector to the list of vectors
        corpus_vectors.append(avg_vector)

    # Convert the list of vectors to a 2D array and return it
    return np.array(corpus_vectors)


def vectorizer_w2v_v2(corpus, model, num_features):
    corpus_vectors = []  # Initialize empty list to store the vectors for all documents

    # Iterate over all documents in corpus
    for doc in corpus:
        # Initialize empty array to store the vector representation of the document
        word_vectors = np.zeros((len(doc), num_features))
        # Initialize counter to keep track of the number of words in the document
        word_count = 0
        # Iterate over all words in the document
        for idx, word in enumerate(doc):
            # Check if the word is present in the vocabulary
            if word in model.key_to_index:
                # If yes, then add its vector to the word_vectors array
                word_vectors[idx] = model[word]
                # Increment the counter by 1
                word_count += 1

        # Compute the average by dividing the sum of all word vectors by the number of words
        if word_count == 0:
            avg_vector = np.zeros(
                (num_features,)
            )  # If no word is present in the vocabulary, then return a vector of zeros
        else:
            avg_vector = np.mean(
                word_vectors, axis=0
            )  # Else, return the average vector of the document

        # Append the average vector to the list of vectors
        corpus_vectors.append(avg_vector)

    # Convert the list of vectors to a 2D array and return it
    return np.array(corpus_vectors)    