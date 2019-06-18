import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from keras.models import Model
from keras.layers import Input, Dense, Embedding, SpatialDropout1D, add, concatenate, Reshape,Flatten
from keras.layers import CuDNNLSTM, Bidirectional, GlobalMaxPooling1D, GlobalAveragePooling1D, LSTM,Dropout
from keras.preprocessing import text, sequence
from keras.callbacks import LearningRateScheduler



# Variables

EMBEDDING_FILES = [
    '../input/fasttext-crawl-300d-2m/crawl-300d-2M.vec',
    '../input/glove840b300dtxt/glove.840B.300d.txt'
]

BATCH_SIZE = 512
LSTM_UNITS = 128
MAX_LEN = 220


IDENTITY_COLUMNS = [
    'male', 'female', 'homosexual_gay_or_lesbian', 'christian', 'jewish',
    'muslim', 'black', 'white', 'psychiatric_or_mental_illness'
]
AUX_COLUMNS = ['target', 'severe_toxicity', 'obscene', 'identity_attack', 'insult', 'threat']
TEXT_COLUMN = 'comment_text'
TARGET_COLUMN = 'target'
CHARS_TO_REMOVE = '!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n“”’\'∞θ÷α•à−β∅³π‘₹´°£€\×™√²—'








# =============================================================================
# Load Embeddings
# =============================================================================
def get_coefs(word, *arr):
    return word, np.asarray(arr, dtype='float32')


def load_embeddings(path):
    with open(path) as f:
        return dict(get_coefs(*line.strip().split(' ')) for line in f)


def build_matrix(word_index, path):
    embedding_index = load_embeddings(path)
    embedding_matrix = np.zeros((len(word_index) + 1, 300))
    for word, i in word_index.items():
        try:
            embedding_matrix[i] = embedding_index[word]
        except KeyError:
            pass
    return embedding_matrix



# =============================================================================
# Data
# =============================================================================
train_df = pd.read_csv('../input/jigsaw-unintended-bias-in-toxicity-classification/train.csv')
test_df = pd.read_csv('../input/jigsaw-unintended-bias-in-toxicity-classification/test.csv')


x_train = train_df[TEXT_COLUMN].astype(str)
y_train = train_df[TARGET_COLUMN].values

y_aux_train = train_df[AUX_COLUMNS].values
x_test = test_df[TEXT_COLUMN].astype(str)



# =============================================================================
# Data processing 
# =============================================================================
for column in IDENTITY_COLUMNS + [TARGET_COLUMN]:
    train_df[column] = np.where(train_df[column] >= 0.5, True, False)

tokenizer = text.Tokenizer(filters=CHARS_TO_REMOVE)
tokenizer.fit_on_texts(list(x_train) + list(x_test))

x_train = tokenizer.texts_to_sequences(x_train)
x_test = tokenizer.texts_to_sequences(x_test)

x_train = sequence.pad_sequences(x_train, maxlen=MAX_LEN)
x_test = sequence.pad_sequences(x_test, maxlen=MAX_LEN)

print("---- Weights")
sample_weights = np.ones(len(x_train), dtype=np.float32)
sample_weights += train_df[IDENTITY_COLUMNS].sum(axis=1)
sample_weights += train_df[TARGET_COLUMN] * (~train_df[IDENTITY_COLUMNS]).sum(axis=1)
sample_weights += (~train_df[TARGET_COLUMN]) * train_df[IDENTITY_COLUMNS].sum(axis=1) * 5
sample_weights /= sample_weights.mean()



# =============================================================================
# Model
# =============================================================================
print("---- Carga de Embeddings")

embedding_matrix = np.concatenate(
    [build_matrix(tokenizer.word_index, f) for f in EMBEDDING_FILES], axis=-1)

checkpoint_predictions = []
weights = []


#def build_model(embedding_matrix):
#    input_layer = Input(shape=(None,))
#    x = Embedding(*embedding_matrix.shape, weights=[embedding_matrix], trainable=False)(input_layer)
#    x = SpatialDropout1D(0.2)(x)
#    # x = Dense(256)(x)
#    # x = Flatten()(x)
#    # x = Reshape((-1,600))(x)
#
#    y = Reshape((1,22))(input_layer)
#    y = Bidirectional(CuDNNLSTM(128,return_sequences=True))(y)
#    # y = Flatten()(y)
#    # y = Bidirectional(CuDNNLSTM(LSTM_UNITS, return_sequences=True))(y)
#
#    hidden = concatenate([x,y])
#    # hidden = add([hidden, Dense(DENSE_HIDDEN_UNITS, activation='relu')(hidden)])
#    # hidden = add([hidden, Dense(DENSE_HIDDEN_UNITS, activation='relu')(hidden)])
#    # hidden = Flatten()(hidden)
#    hidden = Dense(256, activation='relu')(hidden)
#    result = Dense(128, activation='relu')(hidden)
#    result = Dense(1, activation='sigmoid')(hidden)
#    
#    model = Model(inputs=input_layer, outputs=result)
#    model.compile(loss='binary_crossentropy', optimizer='adam')
#    model.summary()
#    return model





def build_cpu_model(embedding_matrix):
    input_layer = Input(shape=(220,))
    x = Embedding(328390,600, weights=[embedding_matrix], trainable=False)(input_layer)
    x = Flatten()(x)
#    x = SpatialDropout1D(0.2)(x)
    x = Dropout(0.2)(x)

#    x = Reshape((1,600))(x)
#    x = Dense(256)(x)
    # x = Dense(256)(x)
    # x = Reshape((-1,600))(x)

    y = Reshape((1,220))(input_layer)
    y = Bidirectional(LSTM(128,return_sequences=True))(y)
    y = Flatten()(y)
    # y = Bidirectional(CuDNNLSTM(LSTM_UNITS, return_sequences=True))(y)



    
    hidden = concatenate([x,y],axis=1)
    # hidden = add([hidden, Dense(DENSE_HIDDEN_UNITS, activation='relu')(hidden)])
    # hidden = add([hidden, Dense(DENSE_HIDDEN_UNITS, activation='relu')(hidden)])
    # hidden = Flatten()(hidden)
    hidden = Dense(256, activation='relu')(hidden)
    result = Dense(128, activation='relu')(hidden)
    result = Dense(1, activation='sigmoid')(result)
    
    model = Model(inputs=input_layer, outputs=result)
    model.compile(loss='binary_crossentropy', optimizer='adam')
    model.summary()
    return model



model = build_cpu_model(embedding_matrix)





print("---- Entrenamiento NN")
model.fit(x_train,y_train,
        batch_size=BATCH_SIZE,
        epochs=1,
        verbose=1)
        
        ,
        sample_weight=[sample_weights.values, np.ones_like(sample_weights)],
    )
# checkpoint_predictions.append(model.predict(x_test, batch_size=2048)[0].flatten())
# weights.append(2 ** global_epoch)


predictions = model.predict(x_test, batch_size=2048)[0].flatten()



# =============================================================================
# Send submissions
# =============================================================================
submission = pd.DataFrame.from_dict({
    'id': test_df.id,
    'prediction': predictions
})
submission.to_csv('submission.csv', index=False)