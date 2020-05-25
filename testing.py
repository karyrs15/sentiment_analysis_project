from keras.models import load_model
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
import nltk
import re
import pickle

# download stopwords (we're gonna need it later)
nltk.download('stopwords')
from nltk.corpus import stopwords

# clean text to remove users, links and stopwords and then split it in tokens
def clean_text(text):
    text_cleaning_re = "@\S+|https?:\S+|http?:\S|[^A-Za-z0-9]+"
    text = re.sub(text_cleaning_re, ' ', str(text).lower()).strip()
    
    tokens = text.split()
    stop_words = set(stopwords.words('english'))
    tokens = [w for w in tokens if w not in stop_words]
    
    return " ".join(tokens)

# decode score prediction from the model, to be 0 or 1
def decode_prediction(prediction):
    return 'Negative' if prediction < 0.5 else 'Positive'

max_length = 50

# load model
model = load_model('model_final.h5')
# loading tokenizer
with open('tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)

while True:
    # test model with a new query
    tweet = input("Enter a tweet: ")

    # clean query text
    input_text = clean_text(tweet)
    # tokenize and pad query test as in training
    input_text = pad_sequences(tokenizer.texts_to_sequences([input_text]),
                            maxlen = max_length)

    # get model prediction
    prediction = model.predict([input_text])[0]
    # get decode prediction
    label = decode_prediction(prediction)

    print("Tweet: \n\n{}\n".format(tweet))
    print("Score: {} Label: {}\n".format(prediction, label))