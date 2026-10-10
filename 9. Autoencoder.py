
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.datasets import mnist
import matplotlib.pyplot as plt

# Load MNIST Dataset
(x_train, _), (x_test, _) = mnist.load_data()

# Preprocess Images
x_train = x_train.reshape(60000, 784) / 255.0
x_test = x_test.reshape(10000, 784) / 255.0

# Create Autoencoder Model
input_img = tf.keras.Input(shape=(784,))

# Encoder
encoded = layers.Dense(32, activation='relu')(input_img)

# Decoder
decoded = layers.Dense(784, activation='sigmoid')(encoded)

# Build Autoencoder
autoencoder = models.Model(input_img, decoded)

# Compile Model
autoencoder.compile(
    optimizer='adam',
    loss='binary_crossentropy'
)

# Train Autoencoder
autoencoder.fit(
    x_train,
    x_train,
    epochs=10,
    batch_size=256,
    shuffle=True,
    validation_data=(x_test, x_test)
)

# Reconstruct Images
decoded_imgs = autoencoder.predict(x_test)

# Display Original and Reconstructed Images
n = 10
plt.figure(figsize=(20, 4))

for i in range(n):

    # Original Image
    plt.subplot(2, n, i + 1)
    plt.imshow(
        x_test[i].reshape(28, 28),
        cmap='gray'
    )
    plt.axis('off')

    # Reconstructed Image
    plt.subplot(2, n, i + 1 + n)
    plt.imshow(
        decoded_imgs[i].reshape(28, 28),
        cmap='gray'
    )
    plt.axis('off')

plt.show()
