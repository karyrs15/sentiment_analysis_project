import tensorflow as tf
from keras.models import load_model
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

# decode score prediction from the model, to be 0 or 1
def decode_prediction(prediction):
    return 'Negative' if prediction < 0.5 else 'Positive'

# load model
model = load_model('model_final.h5')
# loading tokenizer
with open('tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)

# test model with a new query
tweet = "im sad"

# clean query text
input_text = setup_file_sentiment_analysis.clean_text(tweet)
# tokenize and pad query test as in training
input_text = pad_sequences(tokenizer.texts_to_sequences([input_text]),
                        maxlen = max_length)

# get model prediction
prediction = model.predict([input_text])[0]
# get decode prediction
label = decode_prediction(prediction)

print("Tweet: \n\n{}\n".format(query_text))
print("Score: {} Label: {}".format(prediction, label))