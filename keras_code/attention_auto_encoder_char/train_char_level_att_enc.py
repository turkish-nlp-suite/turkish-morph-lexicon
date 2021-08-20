from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from keras.models import Sequential, Model
from keras.layers import LSTM, Dense, Bidirectional, Dropout, Input, Embedding, GlobalMaxPool1D, TimeDistributed, RepeatVector, Permute, Flatten, Activation, multiply, Lambda
from keras import optimizers
import numpy as np
from keras.callbacks import TensorBoard
from keras.layers import Lambda
import keras
import os, json, codecs


alphabet = "abcçdefghıijklmnoöprsştuüvyzwqx'_-0123456789"
tokenizer = Tokenizer(char_level=True, filters=None, lower=True)
tokenizer.fit_on_texts(alphabet)


an_tokenizer = Tokenizer(lower=True)
analysis_list = open("ana.txt").read().split()
analysis_list = filter(None, analysis_list)
an_tokenizer.fit_on_texts(analysis_list)

tokenizer_json = tokenizer.to_json()
with codecs.open('char_tokenizer.json', 'w', encoding='utf-8') as f:
    f.write(json.dumps(tokenizer_json, ensure_ascii=False))


an_tokenizer_json = an_tokenizer.to_json()
with codecs.open('analysis_tokenizer.json', 'w', encoding='utf-8') as f:
    f.write(json.dumps(an_tokenizer_json, ensure_ascii=False))

MAX_LEN = 50
OUT_LEN = 20


embed_size = 100
tensorboard_callback = TensorBoard(log_dir="charAutoenc", histogram_freq=1, write_graph=True, write_images=True, embeddings_freq=1)



word_in = Input(shape=(MAX_LEN,))
x =  Embedding(input_dim = len(tokenizer.word_index)+1, output_dim = 100, input_length=MAX_LEN)(word_in)
lstm = LSTM(units=100, return_sequences=True)(x)  # variational biLSTM
attention = Dense(1, activation='tanh')(lstm)
attention = Flatten()(attention)
attention = keras.layers.Activation('softmax', name='attention_vec')(attention)
attention = RepeatVector(100)(attention)
attention = Permute([2, 1])(attention)

combined = multiply([lstm, attention])
combined_mul = Flatten()(combined)


rv = RepeatVector(30)(combined_mul)
lstmd = LSTM(units=300, return_sequences=True)(rv)
lemmass = Lambda(lambda x: x[:,:20,] )(lstmd)
analysiss = Lambda(lambda x: x[:,20:,:])(lstmd)

lemma_output = TimeDistributed(Dense(45, activation="softmax"))(lemmass)
analysis_output = TimeDistributed(Dense(127, activation="softmax"))(analysiss)

model = Model(inputs=word_in, outputs=[lemma_output, analysis_output])

keras.utils.plot_model(model, to_file="model.png", show_shapes=True)

model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["sparse_categorical_accuracy"])



def iterate_one_file(file_name):
    sentences = []
    analysis = []
    lemmas = []

    with open(file_name, "r", encoding="utf-8") as file:
        for line in file:
            l = line.strip()
            word, res = l.split(" ")
            word = word.strip()
            lemma, ana = res.strip().split(",", 1)
            analysis.append(ana.split(","))
            sentences.append(word)
            lemmas.append(lemma)


    sequences = tokenizer.texts_to_sequences(sentences)
    lemma_seqs = tokenizer.texts_to_sequences(lemmas)
    anl_seqs =  an_tokenizer.texts_to_sequences(analysis)

    X_char = pad_sequences(sequences, MAX_LEN, padding="post")
    lemma_char = pad_sequences(lemma_seqs, OUT_LEN, padding="post")
    anl_word = pad_sequences(anl_seqs, 10, padding="post")

    X_char = np.array(X_char)
    lemmas = np.array(lemma_char)
    analysis = np.array(anl_word)

    lemmas = lemmas.reshape(len(lemma_seqs) , OUT_LEN, 1)
    analysis = analysis.reshape(len(anl_seqs), 10, 1)

    history = model.fit(X_char, [lemmas, analysis], batch_size=64, epochs=1, verbose=1, validation_split=0.1)


files = []


base2 = "../../../turkish-morph-dictionaries/lemmas"
dirs2 = ["alpha", "adjective", "geopraphical_name", "noun", "poss_noun", "pronoun", "proper_noun", "verb"]
for dirn in dirs2:
    path = os.path.join(base2, dirn)
    for r, d, f in os.walk(path):
        for filem in f:
            if filem.endswith(".txt"):
                files.append(os.path.join(r, filem))

for filen in files:
        iterate_one_file(filen)
model.save("charAutoEnc.hd5")

def predict(ww):
    tt = tokenizer.texts_to_sequences([ww])
    ptt = pad_sequences(tt, MAX_LEN, padding="post")
    pred = model.predict(ptt)
    lem = pred[0]
    anll = pred[1]
    xx = lem.argmax(axis=-1).tolist()
    chars = tokenizer.sequences_to_texts(xx)
    yy = anll.argmax(axis=-1).tolist()
    als = an_tokenizer.sequences_to_texts(yy)
    print(chars, als)




word1 = "googledayken"
word2 = "twitterdan"
word3 = "twitterdaki"
word4 = "twitterdan"
word5= "twitterdayken"
word6 = "googledan"
word7 = "googleken"
word8 = "evden"
word9 = "evdeyken"
word10 =  "evdyken"
word11 = "bndeyken"
word12 = "sinirliyseniz"
word13 = "okuldayken"
word14 = "okldayken"
word15 = "okuldaykn"
word16 = "okldaykn"
word17 = "iyiyse"



for ww in [word1, word2, word3, word4, word5, word6, word7, word8, word9, word10, word11, word12, word13, word14, word15, word16, word17]:
    predict(ww)
