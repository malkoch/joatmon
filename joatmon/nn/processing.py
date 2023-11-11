import numpy as np


class OneHotEncoder:
    def __init__(self):
        self._data = {}
        self._reverse_data = {}

        self._eye = None

    def fit(self, data):
        length = len(data)

        self._eye = np.eye(length)

        for idx, row in enumerate(data):
            if row not in self._data:
                self._data[row] = self._eye[idx].tolist()
            if idx not in self._reverse_data:
                self._reverse_data[idx] = row

    def transform(self, data):
        return [self._data[row] for row in data]

    def inverse_transform(self, data):
        return [self._reverse_data[self._eye.tolist().index(row)] for row in data]


class Tokenizer:
    def __init__(self, num_words=1000):
        self.num_words = num_words
        self.document_count = 0
        self.word_counts = {}
        self.word_docs = {}
        self.oov_token = None
        self.word_index = {}
        self.index_word = {}
        self.index_docs = {}

    def fit_on_text(self, data):
        for text in data:
            self.document_count += 1
            for w in text:
                if w in self.word_counts:
                    self.word_counts[w] += 1
                else:
                    self.word_counts[w] = 1
            for w in set(text):
                # In how many documents each word occurs
                if w in self.word_docs:
                    self.word_docs[w] += 1
                else:
                    self.word_docs[w] = 1

        wcounts = list(self.word_counts.items())
        wcounts.sort(key=lambda x: x[1], reverse=True)
        # forcing the oov_token to index 1 if it exists
        if self.oov_token is None:
            sorted_voc = []
        else:
            sorted_voc = [self.oov_token]
        sorted_voc.extend(wc[0] for wc in wcounts)

        # note that index 0 is reserved, never assigned to an existing word
        self.word_index = dict(
            zip(sorted_voc, list(range(1, len(sorted_voc) + 1)))
        )

        self.index_word = {c: w for w, c in self.word_index.items()}

        for w, c in list(self.word_docs.items()):
            self.index_docs[self.word_index[w]] = c

    def text_to_sequence(self, data):
        num_words = self.num_words
        oov_token_index = self.word_index.get(self.oov_token)
        for text in data:
            vect = []
            for w in text:
                i = self.word_index.get(w)
                if i is not None:
                    if num_words and i >= num_words:
                        if oov_token_index is not None:
                            vect.append(oov_token_index)
                    else:
                        vect.append(i)
                elif self.oov_token is not None:
                    vect.append(oov_token_index)
            yield vect

    def sequence_to_text(self, data):
        num_words = self.num_words
        oov_token_index = self.word_index.get(self.oov_token)
        for seq in data:
            vect = []
            for num in seq:
                word = self.index_word.get(num)
                if word is not None:
                    if num_words and num >= num_words:
                        if oov_token_index is not None:
                            vect.append(self.index_word[oov_token_index])
                    else:
                        vect.append(word)
                elif self.oov_token is not None:
                    vect.append(self.index_word[oov_token_index])
            vect = " ".join(vect)
            yield vect


def word_tokenize(data):
    return data.split(' ')


def pad_sequences(
        sequences,
        maxlen=None,
        dtype="int32",
        padding="pre",
        truncating="pre",
        value=0.0,
):
    num_samples = len(sequences)

    lengths = []
    sample_shape = np.asarray(sequences[0]).shape[1:]

    for x in sequences:
        lengths.append(len(x))

    if maxlen is None:
        maxlen = np.max(lengths)

    x = np.full((num_samples, maxlen) + sample_shape, value, dtype=dtype)
    for idx, s in enumerate(sequences):
        if not len(s):
            continue  # empty list/array was found
        if truncating == "pre":
            trunc = s[-maxlen:]
        elif truncating == "post":
            trunc = s[:maxlen]
        else:
            raise ValueError(f'Truncating type "{truncating}" not understood')

        # check `trunc` has expected shape
        trunc = np.asarray(trunc, dtype=dtype)
        if trunc.shape[1:] != sample_shape:
            raise ValueError(
                f"Shape of sample {trunc.shape[1:]} of sequence at "
                f"position {idx} is different from expected shape "
                f"{sample_shape}"
            )

        if padding == "post":
            x[idx, : len(trunc)] = trunc
        elif padding == "pre":
            x[idx, -len(trunc):] = trunc
        else:
            raise ValueError(f'Padding type "{padding}" not understood')
    return x
