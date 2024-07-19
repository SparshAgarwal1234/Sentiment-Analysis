import pandas as pd

df = pd.read_csv('/content/tripadvisor_hotel_reviews.csv')
print(df.head())

df = df[['Review', 'Rating']]
df['sentiment'] = df['Rating'].apply(lambda x: 'positive' if x > 3 else 'negative' if x < 3 else 'neutral')
df = df[['Review', 'sentiment']]
df = df.sample(frac=1).reset_index(drop=True)

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

tokenizer = Tokenizer(num_words=5000, oov_token='<OOV>')
tokenizer.fit_on_texts(df['Review'])
word_index = tokenizer.word_index
sequences = tokenizer.texts_to_sequences(df['Review'])
padded_sequences = pad_sequences(sequences, maxlen=100, truncating='post')

sentiment_labels = pd.get_dummies(df['sentiment']).values

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(padded_sequences, sentiment_labels, test_size=0.2, random_state=42)

from keras.models import Sequential
from keras.layers import Embedding, Dense, Dropout

model = Sequential()
model.add(Embedding(input_dim=5000, output_dim=128, input_length=100))
model.add(Dropout(0.2))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(3, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

model.fit(x_train, y_train, epochs=10, batch_size=32, validation_data=(x_test, y_test))

y_pred = np.argmax(model.predict(x_test), axis=-1)
print("Accuracy:", accuracy_score(np.argmax(y_test, axis=-1), y_pred))

model.save('sentiment_analysis_model.h5')
with open('tokenizer.pickle', 'wb') as handle:
    pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)

    def predict_sentiment(text):
      text_sequence = tokenizer.texts_to_sequences([text])
      text_sequence = pad_sequences(text_sequence, maxlen=100)
      predicted_rating = model.predict(text_sequence)[0]
      if np.argmax(predicted_rating) == 0:
        return 'Negative'
      elif np.argmax(predicted_rating) == 1:
        return 'Neutral'
      else:
        return 'Positive'