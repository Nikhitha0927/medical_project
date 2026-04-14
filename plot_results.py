import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import tensorflow as tf

# load model
model = tf.keras.models.load_model("medical_model.h5")

# load dataset
test_ds = tf.keras.preprocessing.image_dataset_from_directory(
    "dataset/test",
    image_size=(150,150),
    batch_size=32,
    shuffle=False
)

# true labels
y_true = np.concatenate([y for x, y in test_ds], axis=0)

# predictions
y_pred = np.argmax(model.predict(test_ds), axis=1)

# confusion matrix
cm = confusion_matrix(y_true, y_pred)

# plot
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot()
plt.show()