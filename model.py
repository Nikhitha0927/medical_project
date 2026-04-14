import tensorflow as tf
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt

img_size = 150

train_data = tf.keras.preprocessing.image_dataset_from_directory(
    'dataset/train',
    image_size=(img_size, img_size),
    batch_size=32
)

test_data = tf.keras.preprocessing.image_dataset_from_directory(
    'dataset/test',
    image_size=(img_size, img_size),
    batch_size=32
)

normalization_layer = layers.Rescaling(1./255)
train_data = train_data.map(lambda x, y: (normalization_layer(x), y))
test_data = test_data.map(lambda x, y: (normalization_layer(x), y))

model = models.Sequential([
    layers.Conv2D(32, (3,3), activation='relu', input_shape=(150,150,3)),
    layers.MaxPooling2D(),

    layers.Conv2D(64, (3,3), activation='relu'),
    layers.MaxPooling2D(),

    layers.Conv2D(128, (3,3), activation='relu'),
    layers.MaxPooling2D(),

    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(2, activation='softmax')
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

history = model.fit(train_data, epochs=5, validation_data=test_data)

model.save("medical_model.h5")

plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title("Accuracy")
plt.show()

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title("Loss")
plt.show()