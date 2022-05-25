import string
import numpy as np
from keras import optimizers
from keras.models import Sequential
from keras.models import load_model
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Dense, LSTM, Embedding, RepeatVector, Dropout
from sklearn.model_selection import train_test_split
from tensorflow import optimizers
import matplotlib.pyplot as plt


def read_text(filename):
    with open(filename, mode='rt', encoding='utf-8') as file:
        text = file.read()
        sentences = text.strip().split('\n')
        return [sentence.split('\t') for sentence in sentences]  # Возвращает [[en, de],[en, de],[en, de],...]


print("""\n========================== ЗАГРУЗКА ДАННЫХ ==========================""")
count_words = 30000  # будет протестировано 1k / 10k / 30k / 100k фраз
dataSet = read_text("./data/DataSet_En_De_language.txt")
dataSetArray = np.array(dataSet)[:count_words, :]  # используется 30к фраз
print(">> Размер словаря:", dataSetArray.shape)


print("""\n===================== ПРЕДВАРИТЕЛЬНАЯ ОБРАБОТКА =====================""")
# Удаление пунктуации для каждого столбца (языка)
dataSetArray[:, 0] = [s.translate(str.maketrans('', '', string.punctuation)) for s in dataSetArray[:, 0]]
dataSetArray[:, 1] = [s.translate(str.maketrans('', '', string.punctuation)) for s in dataSetArray[:, 1]]
print(">> 25%")

# Перевод всех фраз в нижний регистр
for i in range(len(dataSetArray)):
    dataSetArray[i, 0] = dataSetArray[i, 0].lower()
    dataSetArray[i, 1] = dataSetArray[i, 1].lower()
print(">> 50%")

# Tokenizer заменяет слова в предложениях их цифровыми кодами.
enTokenizer = Tokenizer()  # English tokenizer
enTokenizer.fit_on_texts(dataSetArray[:, 0])
enSize = len(enTokenizer.word_index) + 1
enLength = 8
print(">> 75%")

deTokenizer = Tokenizer()  # Deutsch tokenizer
deTokenizer.fit_on_texts(dataSetArray[:, 1])
deSize = len(deTokenizer.word_index) + 1
deLength = 8
print(">> 100%")
print(">> Tokenizer заменил слова в предложениях их цифровыми кодами.")


print("""\n====================== ОБУЧЕНИЕ НЕЙРОННОЙ СЕТИ ======================""")

def encode_sequences(tokenizer, length, lines):
    # integer encode sequences
    seq = tokenizer.texts_to_sequences(lines)
    # pad sequences with 0 values
    seq = pad_sequences(seq, maxlen=length, padding='post')
    return seq


# Разделение данных на обучающий и тестовый наборы
train, test = train_test_split(dataSetArray, test_size=0.2, random_state=12)

# Подготовка обучающих данных
trainX = encode_sequences(enTokenizer, enLength, train[:, 0])
trainY = encode_sequences(deTokenizer, deLength, train[:, 1])

# Подготовка тестовых данных
testX = encode_sequences(enTokenizer, enLength, test[:, 0])
testY = encode_sequences(deTokenizer, deLength, test[:, 1])


# Построение модели
def makeModel(in_vocab, out_vocab, in_timesteps, out_timesteps, n):
    model = Sequential()
    model.add(Embedding(in_vocab, n, input_length=in_timesteps, mask_zero=True))
    model.add(LSTM(n))
    model.add(Dropout(0.3))
    model.add(RepeatVector(out_timesteps))
    model.add(LSTM(n, return_sequences=True))
    model.add(Dropout(0.3))
    model.add(Dense(out_vocab, activation='softmax'))
    model.compile(optimizer=optimizers.RMSprop(learning_rate=0.001), loss='sparse_categorical_crossentropy')
    return model


print(f'>> Немецких слов: {deSize}')
print(f'>> Английских слов: {enSize}')
print('\n')

# Компиляция модели (с 512 скрытыми единицами)
model = makeModel(enSize, deSize, enLength, deLength, 512)

# Тренировка модели
epochsNum = 250
history = model.fit(trainX, trainY.reshape(trainY.shape[0], trainY.shape[1], 1),
                    epochs=epochsNum,
                    batch_size=512,
                    validation_split=0.2,
                    callbacks=None,
                    verbose=1)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.legend(['train', 'validation'])
plt.show()
model.save('en-de-model.h5')

# Загрузка модели
model = load_model('en-de-model.h5')


# get_word используется для обратного преобразования слов в числа.
def getWord(n, tokenizer):
    if n == 0:
        return ""
    for word, index in tokenizer.word_index.items():
        if index == n:
            return word
    return ""


# Входные данные
inputSentences = ["the weather is nice today",
                  "my name is tom",
                  "how old are you",
                  "where is the nearest shop"]
inputData = encode_sequences(enTokenizer, enLength, inputSentences)
print('\n')
print(">> inputSentences:", inputSentences)
print(">> inputData:", inputData.shape)

# Функцией predict необходима для получения перевода
result = model.predict(inputData)
result = np.argmax(result, axis=1)


print("""\n===================== РЕЗУЛЬТАТ =====================""")
def printResult(inputSentence, translated):
    print(f'>> Английский: {inputSentence}')

    translated_text = ''
    for word in translated:
        translated_text += word + ' '
    print(f'>> Немецкий: ' + translated_text)

    print('\n')


print(">> Predict:", result.shape, '\n')

# 1
translated = [getWord(result[0][0], deTokenizer), getWord(result[0][1], deTokenizer),
              getWord(result[0][2], deTokenizer),
              getWord(result[0][3], deTokenizer)]
printResult(inputSentences[0], translated)

# 2
translated = [getWord(result[1][0], deTokenizer), getWord(result[1][1], deTokenizer),
              getWord(result[1][2], deTokenizer),
              getWord(result[1][3], deTokenizer)]
printResult(inputSentences[1], translated)

# 3
translated = [getWord(result[2][0], deTokenizer), getWord(result[2][1], deTokenizer),
              getWord(result[2][2], deTokenizer),
              getWord(result[2][3], deTokenizer)]
printResult(inputSentences[2], translated)

# 4
translated = [getWord(result[3][0], deTokenizer), getWord(result[3][1], deTokenizer),
              getWord(result[3][2], deTokenizer),
              getWord(result[3][3], deTokenizer)]
printResult(inputSentences[3], translated)
