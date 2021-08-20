from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from keras.models import Sequential, Model
from keras.layers import LSTM, Dense, Bidirectional, Dropout, Input, Embedding, GlobalMaxPool1D, TimeDistributed, RepeatVector
from keras import optimizers
import numpy as np
from keras.callbacks import TensorBoard
from keras.layers import Activation, concatenate, dot, Concatenate
import os,json,codecs


alphabet = "abcçdefghıijklmnoöprsştuüvyzwqx'_!-0123456789"
tokenizer = Tokenizer(char_level=True, filters=None, lower=True)
tokenizer.fit_on_texts(alphabet)


tokenizer_json = tokenizer.to_json()
with codecs.open('tokenizer.json', 'w', encoding='utf-8') as f:
    f.write(json.dumps(tokenizer_json, ensure_ascii=False))


MAX_LEN = 50
OUT_LEN = 30
embed_size = 100

encoder_input = Input(shape=(None,))
encoder_embedding =  Embedding(input_dim = len(tokenizer.word_index)+1, output_dim = 100)
encoder_embed = encoder_embedding(encoder_input)
encoder_LSTM = LSTM(units=256, return_state=True, return_sequences=True)
encoder_outputs, encoder_h, encoder_c = encoder_LSTM(encoder_embed)  # variational biLSTM
encoder_states = [encoder_h, encoder_c]
decoder_input = Input(shape=(None,))
decoder_embed = encoder_embedding(decoder_input)
decoder_LSTM = LSTM(256,return_sequences=True, return_state = True)
decoder_outputs, state_h, state_c = decoder_LSTM(decoder_embed, initial_state=encoder_states)
attention = dot([decoder_outputs, encoder_outputs], axes=[2, 2])
attention = Activation('softmax', name="attention_vec")(attention)
context = dot([attention, encoder_outputs], axes=[2,1])
decoder_outputs = concatenate([context, decoder_outputs])
decoder_dense = Dense(101,activation='softmax')
decoder_outputs = decoder_dense(decoder_outputs)
model = Model(inputs=[encoder_input, decoder_input],outputs=[decoder_outputs])
model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["sparse_categorical_accuracy"])


def iterate_one_file(file_name):
    words = []
    lemma_analysis = []

    with open(file_name, "r", encoding="utf-8") as file:
        for line in file:
            l = line.strip()
            word, res = l.split(" ")
            word = word.strip()
            lemma, ana = res.strip().split(",", 1)
            words.append(list(word))
            lemma_analysis.append(["!"] + list(lemma) + ana.split(",") + ["_"])


    sequences = tokenizer.texts_to_sequences(words)
    lemma_seqs = tokenizer.texts_to_sequences(lemma_analysis)


    target = [lemma_seq[1:] for lemma_seq in lemma_analysis]
    target_seqs = tokenizer.texts_to_sequences(target)

    X_char = pad_sequences(sequences, MAX_LEN, padding="post")
    lemma_char = pad_sequences(lemma_seqs, OUT_LEN, padding="post")
    target_char = pad_sequences(target_seqs, OUT_LEN, padding="post")

    X_char = np.array(X_char)
    y = np.array(lemma_char)

    target_data = np.array(target_char)
    target_data = target_data.reshape(len(words) , OUT_LEN, 1)

    model.fit(x=[X_char,y],
          y=target_data,
          batch_size=64,
          epochs=1)

files = []
base2 = "../../../turkish-morph-dictionaries/lemmas"
dirs2 = ["alpha", "adjective", "geopraphical_name", "noun", "poss_noun", "pronoun", "proper_noun", "verb"]
for dirn in dirs2:
    path = os.path.join(base2, dirn)
    for r, d, f in os.walk(path):
        for filem in f:
            if filem.endswith(".txt"):
                files.append(os.path.join(r, filem))

base1 = "../../../turkish-morph-dictionaries/analyses"
dirs1 = ["adjective", "adjective/extra", "adjective/ortac", "adverb",  "conjunct",  "interjection", "noun", "noun/fiilden",   "postposition", "pronoun",  "verb"]
for dirn in dirs1:
    path = os.path.join(base1, dirn)
    for r, d, f in os.walk(path):
        for filem in f:
            if filem.endswith(".txt"):
                files.append(os.path.join(r, filem))

for filen in files:
    with open("completed.txt", "a") as myfile:
        myfile.write(filen+"\n")
    iterate_one_file(filen)
    model.save("charAutoEnc.hd5")






encoder_inf_in = Input(shape=(None,))
encoder_embed_inf = encoder_embedding(encoder_inf_in)
encoder_inf_out, encoder_inf_h, encoder_inf_c = encoder_LSTM(encoder_embed_inf)
encoder_model_inf = Model(inputs=encoder_inf_in, outputs=[encoder_inf_out, encoder_inf_h, encoder_inf_c])

# Decoder inference model
decoder_state_input_h = Input(shape=(256,))
decoder_state_input_c = Input(shape=(256,))
decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]
decoder_hidden_state_input = Input(shape=(None, 256))

dec_emb2 = encoder_embedding(decoder_input)

decoder_out, decoder_h, decoder_c = decoder_LSTM(dec_emb2, initial_state=decoder_states_inputs)

att_inf = dot([decoder_out, decoder_hidden_state_input], axes=[2,2])
att_inf = Activation('softmax')(att_inf)
context_inf = dot([att_inf, decoder_hidden_state_input], axes=[2,1])
decoder_concat_inf = concatenate([context_inf, decoder_out])

decoder_states = [decoder_h , decoder_c]
decoder_out = decoder_dense(decoder_concat_inf)

decoder_model_inf = Model(inputs=[decoder_input] + [decoder_hidden_state_input] + decoder_states_inputs,
                          outputs=[decoder_out] +  decoder_states)



reverse_word_map = dict(map(reversed, tokenizer.word_index.items()))


def decode_sequence(input_seq):
    # Encode the input as state vectors.
    enc_outs, e_h, e_c = encoder_model_inf.predict(input_seq)

    # Generate empty target sequence of length 1.
    target_seq = np.zeros((1,1))

    # Populate the first character of target sequence with the start character.
    target_seq[0, 0] = tokenizer.word_index['!']

    # Sampling loop for a batch of sequences
    # (to simplify, here we assume a batch of size 1).
    stop_condition = False
    decoded_sentence = ''

    while not stop_condition:
        output_tokens, h, c = decoder_model_inf.predict([target_seq] + [enc_outs] + [e_h, e_c])
        # Sample a token
        sampled_token_index = np.argmax(output_tokens[0, -1, :])
        sampled_char = reverse_word_map[sampled_token_index]
        decoded_sentence += ' '+sampled_char

        # Exit condition: either hit max length or find stop token.
        if (sampled_char == '_' or len(decoded_sentence) > 20):
            stop_condition = True

        # Update the target sequence (of length 1).
        target_seq = np.zeros((1,1))
        target_seq[0, 0] = sampled_token_index

        # Update states
        e_h, e_c = h, c

    return decoded_sentence



word1 = "öbürkünde"
word2 = "seninleyken"
word3 = "benimleyken"
word4 = "benimdi"
word5= "kendimizdenmişler"
word6 = "googlelayken"
word7 = "googleken"

for ww in [word1, word2, word3, word4, word5, word6, word7]:
    tt = tokenizer.texts_to_sequences([ww])
    ptt = pad_sequences(tt, MAX_LEN, padding="post")
    pred = decode_sequence(ptt)
    print(ww)
    print("\n")
    print(pred)
    print("-------------------------")


