
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.datasets import imdb
from tensorflow.keras.utils import pad_sequences
import matplotlib.pyplot as plt

# Set Parameters
max_features = 10000
maxlen = 100

# Load IMDB Dataset
(x_train, y_train), (x_test, y_test) = imdb.load_data(
    num_words=max_features
)

# Preprocess Dataset
x_train = pad_sequences(x_train, maxlen=maxlen)
x_test = pad_sequences(x_test, maxlen=maxlen)

# Create LSTM Model
embedding_dim = 128

model = models.Sequential()

model.add(layers.Input(shape=(maxlen,)))
model.add(layers.Embedding(max_features, embedding_dim))
model.add(layers.LSTM(128))
model.add(layers.Dense(1, activation='sigmoid'))

# Compile Model
model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

# Train LSTM Model
history = model.fit(
    x_train,
    y_train,
    epochs=5,
    batch_size=32,
    validation_split=0.1
)

# Evaluate Model
test_loss, test_acc = model.evaluate(x_test, y_test)

print("Test accuracy:", test_acc)

# Training and Validation Loss Graph
plt.figure(figsize=(8, 6))

plt.plot(
    history.history['loss'],
    label='Training Loss'
)

plt.plot(
    history.history['val_loss'],
    label='Validation Loss'
)

plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training and Validation Loss')
plt.legend()
plt.show()

# Training and Validation Accuracy Graph
plt.figure(figsize=(8, 6))

plt.plot(
    history.history['accuracy'],
    label='Training Accuracy'
)

plt.plot(
    history.history['val_accuracy'],
    label='Validation Accuracy'
)

plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Training and Validation Accuracy')
plt.legend()
plt.show()
