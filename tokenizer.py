import tensorflow as tf
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
Sentences = [
    'I Love My Dog',
    'I Love My cat',
    'You Love my dog!', 
    'do you think my dog is amazing?'
]

tokenizer = Tokenizer(num_words= 100, oov_token='<00V>')
tokenizer.fit_on_texts(Sentences)
word_index = tokenizer.word_index
# Try with words that the tokenizer wasn't fit to
test_data = [
    'i really love my dog',
    'my dog loves my manatee'
]

# Generate the sequences
test_seq = tokenizer.texts_to_sequences(test_data)

sequences  = tokenizer.texts_to_sequences(Sentences)
pad_seq = pad_sequences(sequences)
print(word_index)
print(pad_seq)
print(test_seq)