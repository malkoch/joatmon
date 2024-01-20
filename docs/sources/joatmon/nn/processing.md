#


## OneHotEncoder
```python 

```


---
This class is used for performing one-hot encoding on categorical data.


**Attributes**

* **_data** (dict) : A dictionary to store the one-hot encoded representation of each category.
* **_reverse_data** (dict) : A dictionary to store the original category for each one-hot encoded representation.
* **_eye** (numpy.ndarray) : An identity matrix used for one-hot encoding.



**Methods:**


### .fit
```python
.fit(
   data
)
```

---
Fits the encoder to the data.


**Args**

* **data** (list) : The data to fit the encoder to.


### .transform
```python
.transform(
   data
)
```

---
Transforms the data into one-hot encoded format.


**Args**

* **data** (list) : The data to transform.


**Returns**

* **list**  : The one-hot encoded data.


### .inverse_transform
```python
.inverse_transform(
   data
)
```

---
Transforms the one-hot encoded data back into its original format.


**Args**

* **data** (list) : The one-hot encoded data to transform.


**Returns**

* **list**  : The original data.


----


## Tokenizer
```python 
Tokenizer(
   num_words = 1000
)
```


---
This class is used for tokenizing text data.


**Attributes**

* **num_words** (int) : The maximum number of words to keep, based on word frequency.
* **document_count** (int) : The total number of documents (texts) that the tokenizer was trained on.
* **word_counts** (dict) : A dictionary of words and their counts.
* **word_docs** (dict) : A dictionary of words and how many documents each appeared in.
* **oov_token** (str) : The out of vocabulary token.
* **word_index** (dict) : A dictionary of words and their uniquely assigned integers.
* **index_word** (dict) : A dictionary of integers and their corresponding words.
* **index_docs** (dict) : A dictionary of integers and how many documents the corresponding word appeared in.



**Methods:**


### .fit_on_text
```python
.fit_on_text(
   data
)
```

---
Fits the tokenizer to the data.


**Args**

* **data** (list) : The data to fit the tokenizer to.


### .text_to_sequence
```python
.text_to_sequence(
   data
)
```

---
Transforms the data into a sequence of integers.


**Args**

* **data** (list) : The data to transform.


**Yields**

* **list**  : The transformed data.


### .sequence_to_text
```python
.sequence_to_text(
   data
)
```

---
Transforms the sequence of integers back into text.


**Args**

* **data** (list) : The sequence of integers to transform.


**Yields**

* **str**  : The transformed text.


----


### word_tokenize
```python
.word_tokenize(
   data
)
```

---
Tokenizes the text into words.


**Args**

* **data** (str) : The text to tokenize.


**Returns**

* **list**  : The tokenized text.


----


### pad_sequences
```python
.pad_sequences(
   sequences, maxlen = None, dtype = 'int32', padding = 'pre', truncating = 'pre',
   value = 0.0
)
```

---
Pads sequences to the same length.


**Args**

* **sequences** (list) : List of sequences (each sequence is a list of integers).
* **maxlen** (int, optional) : Maximum length for all sequences. If not provided, sequences will be padded to the length of the longest individual sequence.
* **dtype** (str, optional) : Type of the output sequences. Defaults to 'int32'.
* **padding** (str, optional) : 'pre' or 'post': pad either before or after each sequence. Defaults to 'pre'.
* **truncating** (str, optional) : 'pre' or 'post': remove values from sequences larger than `maxlen`, either at the beginning or at the end of the sequences. Defaults to 'pre'.
* **value** (float, optional) : Value to pad the sequences to the desired value. Defaults to 0.0.


**Returns**

* **ndarray**  : Padded sequences.

