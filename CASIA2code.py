import keras
from keras.utils.np_utils import to_categorical # used for converting labels to one-hot-encoding
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D, BatchNormalization
from sklearn.model_selection import train_test_split
from scipy import stats
from sklearn.preprocessing import LabelEncoder
from PIL import Image
import numpy as np
import cv2
from PIL import Image, ImageChops, ImageEnhance
import os
import random
import itertools
from keras.utils.np_utils import to_categorical
from sklearn.utils import shuffle
from keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam


def image_to_ela(path, quality):
    ''' Gets images specified as path and resaves it at a new path resave_path at specified quality'''
    try:
        # Check if the file format is supported (JPEG or PNG)
        if path.endswith('jpg') or path.endswith('jpeg') or path.endswith('png'):
            # Open the image and convert it to RGB mode
            image = Image.open(path).convert('RGB')
            
            # Resave the image with the specified quality
            image.save('resaved.jpg', 'JPEG', quality=quality)
            resaved = Image.open('resaved.jpg')

            # Calculate the ELA (Error Level Analysis) image by taking the difference between the original and resaved image
            ela_image = ImageChops.difference(image, resaved)

            # Get the minimum and maximum pixel values in the ELA image
            band_values = ela_image.getextrema()
            max_value = max([val[1] for val in band_values])

            # If the maximum value is 0, set it to 1 to avoid division by zero
            if max_value == 0:
                max_value = 1

            # Scale the pixel values of the ELA image to the range [0, 255]
            scale = 255.0 / max_value
            ela_image = ImageEnhance.Brightness(ela_image).enhance(scale)

            return ela_image
    except Exception as e:
        print(f'Could not convert {path} to ELA: {str(e)}')

real_image_path = '/kaggle/input/casia-20-image-tampering-detection-dataset/CASIA2/Au/Au_ani_00010.jpg'
Image.open(real_image_path)

image_size = (150, 150)

def prepare_image(image_path):
    return np.array(image_to_ela(image_path, 90).resize(image_size)).flatten() / 255.0

X = [] # ELA converted images
Y = [] # 0 for fake, 1 for real

path = '/kaggle/input/casia-20-image-tampering-detection-dataset/CASIA2/Au'
for dirname, _, filenames in os.walk(path):
    for filename in filenames:
        if filename.endswith('jpg') :
            full_path = os.path.join(dirname, filename)
            X.append(prepare_image(full_path))
            Y.append(1)  
        if len(Y) % 1000 == 0:
            print(f'Processing {len(Y)} images')
#         if len(Y)==100:
#             break
#     if len(Y) == 100:
#         break

# or filename.endswith('png')

random.shuffle(X)
# X = X[:2100]
# Y = Y[:2100]
print(len(X), len(Y))

path = '/kaggle/input/casia-20-image-tampering-detection-dataset/CASIA2/Tp'
for dirname, _, filenames in os.walk(path):
    for filename in filenames:
        if filename.endswith('jpg') :
            full_path = os.path.join(dirname, filename)
            X.append(prepare_image(full_path))
            Y.append(0)
        if len(Y) % 1000 == 0:
            print(f'Processing {len(Y)} images')
#         if len(Y)==200:
#             break
#     if len(Y) == 200:
#         break
# or filename.endswith('png')
print(len(X), len(Y))

for i in range(10):
    X, Y = shuffle(X, Y, random_state=i)

print(Y[0])

print(X[0].reshape(-1,150,150,3))

X = np.array(X)
Y = to_categorical(Y, 2)
X = X.reshape(-1, 150, 150, 3)

print(X[0].shape)
print(Y[0].shape)

from sklearn.model_selection import train_test_split
X_train, X_rest, Y_train, Y_rest = train_test_split(X, Y, test_size=0.2, random_state=42)
del X
del Y

X_test, X_val, Y_test, Y_val = train_test_split(X_rest, Y_rest, test_size=0.2, random_state=42)
del X_rest
del Y_rest

print(len(X_train), len(Y_train))
print(len(X_test), len(Y_test))
print(len(X_val), len(Y_val))

print(X_train[0].shape)
print(Y_train[0].shape)

def build_model():
    model =Sequential()
    model.add(Conv2D(filters = 32, kernel_size = (5, 5), padding = 'valid', activation = 'relu', input_shape = (150, 150, 3)))
    model.add(Conv2D(filters = 32, kernel_size = (5, 5), padding = 'valid', activation = 'relu'))
    model.add(MaxPool2D(pool_size = (2, 2)))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(150, activation = 'relu'))
    model.add(Dropout(0.5))
    model.add(Dense(2, activation = 'sigmoid'))

    
    return model

model = build_model()
model.summary()

batch_size = 8
epochs = 40

from tensorflow.keras.metrics import Precision, Recall
from sklearn.metrics import accuracy_score, f1_score

# model.compile(optimizer='adam',
#               loss='binary_crossentropy',  # For integer labels
#               metrics=['accuracy', Precision(), Recall()])


model.compile(
    optimizer = Adam(learning_rate = 0.0001),
    loss = 'binary_crossentropy',
    metrics=['accuracy', Precision(), Recall()]
)

# model.compile(optimizer = optimizer, loss = 'binary_crossentropy', metrics =['accuracy'])

early_stopping = EarlyStopping(monitor = 'val_acc',
                              min_delta = 0,
                              patience = 2,
                              verbose = 0,
                              mode = 'auto')

history=model.fit(
    X_train, Y_train,
    epochs=epochs,
    batch_size = batch_size,
    validation_data=(X_val,Y_val),
    verbose=1,
# callbacks = [early_stopping]
)

# Evaluate the model on the test dataset
loss, accuracy, precision, recall = model.evaluate(X_test, Y_test)

# Use the model to make predictions
y_pred = model.predict(X_test)

# Convert predicted probabilities to class labels
y_pred_labels = np.argmax(y_pred, axis=1)

# Print the evaluation results
print("Test Loss:", loss)
print("Test Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

import matplotlib.pyplot as plt
plt.plot(history.history['accuracy'])

plt.legend(['accuarcy'])
plt.show()

plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.legend(['accuracy','val_accuracy'])

fig, ax = plt.subplots(2,1)
ax[0].plot(history.history['loss'], color='b', label="Training loss")
ax[0].plot(history.history['val_loss'], color='r', label="validation loss",axes =ax[0])
legend = ax[0].legend(loc='best', shadow=True)

ax[1].plot(history.history['accuracy'], color='b', label="Training accuracy")
ax[1].plot(history.history['val_accuracy'], color='r',label="Validation accuracy")
legend = ax[1].legend(loc='best', shadow=True)


from tensorflow.keras.utils import plot_model
plot_model(model, to_file='model_architecture.png', show_shapes=True)


from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
# Convert predictions to class labels
predicted_labels = np.argmax(y_pred, axis=1)
true_labels = np.argmax(Y_test, axis=1)

# Calculate the confusion matrix
confusion_mtx = confusion_matrix(true_labels, predicted_labels)
# Visualize the confusion matrix using matplotlib
plt.figure(figsize=(8, 6))
plt.imshow(confusion_mtx, interpolation='nearest', cmap=plt.cm.Blues)
plt.title('Confusion Matrix')
plt.colorbar()
plt.xticks(np.arange(1), [str(i) for i in range(1)])
plt.yticks(np.arange(1), [str(i) for i in range(1)])
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.show()

from tensorflow.keras.models import save_model

# Save the model to an h5 file
model.save("myModel2.h5")


json_file = model.to_json()
with open('./mymodel.json', "w") as file:
   file.write(json_file)


