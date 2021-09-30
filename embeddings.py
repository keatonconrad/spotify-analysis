from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np
import boto3

s3 = boto3.resource('s3')


class EmbeddingGenerator:

    def __init__(self, max_sequence_len=50, texts=None, filename=None,
                 embedding_s3_bucket=None, embedding_s3_key=None, num_words=None,
                 char_level=False):
        self.word_index = {}
        self.embeddings_index = {}
        self.embedding_matrix = {}
        self.vocab_size = 0
        self.embedding_dimension = 0
        self.max_sequence_len = max_sequence_len
        self.tokenizer = Tokenizer(
            num_words=num_words,
            filters='\t\n',
            char_level=char_level,
            lower=False,
            oov_token='<unknown>'  # Sets words it doesn't know to this value
        )
        if texts is not None:
            self.generate_word_index(texts)
        if filename or (embedding_s3_bucket and embedding_s3_key):
            self.load_pretrained_embedding(filename, embedding_s3_bucket,
                                           embedding_s3_key)
            if texts is not None:
                self.generate_embedding_matrix()

    def generate_word_index(self, texts):
        """
        Creates the word index from the given texts.

        Args:
            texts: Array of strings
        Returns:
        The generated word index
        """

        self.tokenizer.fit_on_texts(texts)
        self.word_index = self.tokenizer.word_index
        self.vocab_size = len(self.word_index) + 1
        return self.word_index

    def generate_sequences(self, texts):
        """
        Transforms texts into sequences of word indices.
        Pads sequences so that they have equal length.
        Only callable after word index has been generated.

        Args:
            texts: Array of strings
        Returns:
            The padded sequences as a 2D-array of strings
        """

        sequences = self.tokenizer.texts_to_sequences(texts)
        padded = pad_sequences(sequences, maxlen=self.max_sequence_len,
                               padding='post', truncating='post')
        return padded

    def load_pretrained_embedding(self, filename=None, s3_bucket=None, s3_key=None):
        """
        Loads a pretrained embeddings index from the given file.
        Assumes a file with one line per embedding, starting with word and
        followed by coefficients (separated by spaces).

        Args:
            filename: Path to embedding
            s3_bucket: S3 Bucket where embedding is stored
            s3_key: Key of embedding file stored on S3
        Returns:
            The loaded embeddings index
        """

        coefs = []

        if filename:
            with open(filename, encoding='utf-8') as f:
                for line in f:
                    try:
                        values = line.split()
                        word = values[0]
                        coefs = np.asarray(values[1:], dtype='float32')
                        self.embeddings_index[word] = coefs
                    except ValueError: # Handles weird error with glove vectors
                        continue
                self.embedding_dimension = len(coefs)
            return self.embeddings_index

        else:
            obj = s3.Object(s3_bucket, s3_key)
            for line in obj.get()['Body'].iter_lines():
                values = line.decode('utf-8').split()
                word = values[0]
                coefs = np.asarray(values[1:], dtype='float32')
                self.embeddings_index[word] = coefs
            self.embedding_dimension = len(coefs)
            return self.embeddings_index

    def generate_embedding_matrix(self):
        """
        Creates embedding matrix from word and embeddings indices.

        Returns:
            Generated embedding matrix as 2D numpy array
        """

        embedding_matrix = np.zeros((len(self.word_index) + 1, self.embedding_dimension))
        for word, i in self.word_index.items():
            embedding_vector = self.embeddings_index.get(word)
            if embedding_vector is not None and len(embedding_vector) == self.embedding_dimension:
                # Words not found will stay all-zeros
                embedding_matrix[i] = embedding_vector
        self.embedding_matrix = embedding_matrix
        return self.embedding_matrix
