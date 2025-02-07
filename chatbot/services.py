import json, os
import numpy as np
import nltk
import random
import pickle
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.regularizers import l2
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

# Define the path to the static folder
project_folder = os.path.dirname(os.path.dirname(__file__))
static_folder = os.path.join(project_folder, 'staticfiles')

# Ensure the static folder exists
os.makedirs(static_folder, exist_ok=True)


data = []
words = []
classes = []
documents = []
ignore_words = ["?", "!"]
lemmatizer = WordNetLemmatizer()

train_x = []
train_y = []

def load_data():
    data_path = os.path.join(static_folder, 'data.json')
    with open(data_path, 'r') as file:
        data = json.load(file)
        
    return data

def load_words():
    words_path = os.path.join(static_folder, "words.pkl")
    if os.path.exists(words_path):
        return pickle.load(open(words_path, "rb"))
    else:
        raise FileNotFoundError("Words file 'words.pkl' not found. Please train the chatbot first.")
    
def load_classes():
    classes_path = os.path.join(static_folder, "classes.pkl")
    if os.path.exists(classes_path):
        return pickle.load(open(classes_path, "rb"))
    else:
        raise FileNotFoundError("Classes file 'classes.pkl' not found. Please train the chatbot first.")
    
def load_trained_model():
    model_path = os.path.join(static_folder, "chatbot_model.keras")
    if os.path.exists(model_path):
        return load_model(model_path)
    else:
        raise FileNotFoundError("Model file 'chatbot_model.keras' not found. Please train the chatbot first.")

def setup_nltk():
    nltk_data_path = os.path.join(os.path.expanduser("~"), "nltk_data")
    if not os.path.exists(nltk_data_path):
        os.makedirs(nltk_data_path)
    
    if not os.path.exists(os.path.join(nltk_data_path, "tokenizers/punkt")):
        nltk.download("punkt", download_dir=nltk_data_path)
    if not os.path.exists(os.path.join(nltk_data_path, "corpora/wordnet")):
        nltk.download("wordnet", download_dir=nltk_data_path)
    if not os.path.exists(os.path.join(nltk_data_path, "tokenizers/punkt_tab")):
        nltk.download('punkt_tab', download_dir=nltk_data_path)
    
def setup_preprocessing_of_data():
    global data
    data = load_data()
    # Step 2: Preprocessing
    def preprocess_data():
        global words, classes, documents
        for intent in data["intents"]:
            for pattern in intent["patterns"]:
                word_list = word_tokenize(pattern)
                words.extend(word_list)
                documents.append((word_list, intent["tag"]))
                if intent["tag"] not in classes:
                    classes.append(intent["tag"])

        words = [lemmatizer.lemmatize(w.lower()) for w in words if w not in ignore_words]
        
        words = sorted(set(words))
        classes = sorted(set(classes))
    
    # Step 3: Save Preprocessed Data
    def save_preprocessed_data():
        preprocess_data()
        
        # Save words and classes
        words_path = os.path.join(static_folder, "words.pkl")
        classes_path = os.path.join(static_folder, "classes.pkl")
        
        with open(words_path, "wb") as words_file:
            pickle.dump(words, words_file)
        
        with open(classes_path, "wb") as classes_file:
            pickle.dump(classes, classes_file)
        
    save_preprocessed_data()

def setup_training_data():
    global train_x, train_y, documents, lemmatizer
    training = []
    output_empty = [0] * len(classes)

    for doc in documents:
        bag = []
        word_patterns = doc[0]
        word_patterns = [lemmatizer.lemmatize(w.lower()) for w in word_patterns]
        for w in words:
            bag.append(1) if w in word_patterns else bag.append(0)

        output_row = list(output_empty)
        output_row[classes.index(doc[1])] = 1
        training.append([bag, output_row])

    random.shuffle(training)
    training = np.array(training, dtype=object)

    train_x = np.array(list(training[:, 0]))
    train_y = np.array(list(training[:, 1]))
    
def initialize_preprocessed_data():
    setup_nltk()
    setup_preprocessing_of_data()

def initialize_training_data():
    setup_training_data()

def build_model():
    model = Sequential()
    model.add(Dense(258, input_shape=(len(train_x[0]),), activation="relu"))
    model.add(Dropout(0.5))
    model.add(Dense(128, activation="relu"))
    model.add(Dropout(0.5))
    model.add(Dense(64, activation="relu"))
    model.add(Dropout(0.5))
    model.add(Dense(len(train_y[0]), activation="softmax"))

    sgd = SGD(learning_rate=0.01, momentum=0.9, nesterov=True)
    model.compile(loss="categorical_crossentropy", optimizer=sgd, metrics=["accuracy"])
    
    return model

def train_model():
    model_path = os.path.join(static_folder, "chatbot_model.keras")
    if os.path.exists(model_path):
        model = load_model(model_path)
        model.fit(train_x, train_y, epochs=512, batch_size=64, verbose=1)
        model.save(os.path.join(static_folder, "chatbot_model.keras"))
    else:
        model = build_model()
        model.fit(train_x, train_y, epochs=1024, batch_size=64, verbose=1)
        model.save(os.path.join(static_folder, "chatbot_model.keras"))
    
def start_training_process():
    initialize_preprocessed_data()
    initialize_training_data()
    train_model()
    
