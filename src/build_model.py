import cv2
import numpy as np
import os
import sys

from keras import backend as K
from keras.layers import Activation, Add, BatchNormalization, Bidirectional
from keras.layers import Conv2D, Dense, Input, LSTM, MaxPooling2D, Reshape
from keras.layers import TimeDistributed
from keras.models import Model, Sequential

from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split

data_input = "wl1"

image_paths = [data_input + "/{0}".format(f)
                for f in os.listdir(data_input)
                    if os.path.isfile(os.path.join(data_input, f))]

data = []
labels = []

# Load images, convert to greyscale
for image_path in image_paths:
    name = os.path.basename(image_path)
    name = os.path.splitext(name)[0]

    # Load image
    image = cv2.imread(image_path)
    if image is None:
        print("Warning: Skipping file at " + image_path)
        continue

    # Resize
    image = cv2.resize(image, (64, 64))

    # Convert to greyscale
    grey = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    # Convert to float and normalize
    grey = grey.astype('float32')
    grey /= 255

    label = name[0]
    data.append(grey)
    labels.append(label)

alphabet = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M',
            'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z',
            'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm',
            'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z']

# Create numpy arrays
data = np.array(data)
labels = np.array(labels)

# Create training and test data
(X_train, X_test, Y_train, Y_test) = train_test_split(data, labels,
                                        test_size=0.25, random_state=0)

# Convert labels into one-hot encodings for Keras
lb = LabelBinarizer().fit(alphabet)
Y_train = lb.transform(Y_train)
Y_test = lb.transform(Y_test)

X_train = X_train.reshape(X_train.shape[0], 64, 64, 1)
X_test = X_test.reshape(X_test.shape[0], 64, 64, 1)


model = Sequential()

# First convolutional layer
print("Adding first Conv2D")
model.add(Conv2D(filters = 64, kernel_size = (3, 3), padding = "same",
                 activation = "relu", input_shape = (64, 64, 1)))
model.add(MaxPooling2D(pool_size = (2, 2), strides = (2, 2)))

# Second convolutional layer
print("Adding second Conv2D")
model.add(Conv2D(filters = 128, kernel_size = (3, 3), padding = "same",
                 activation = "relu"))
model.add(MaxPooling2D(pool_size = (2, 2), strides = (2, 2)))

# Third convolutional layer
print("Adding third Conv2D")
model.add(Conv2D(filters = 256, kernel_size = (3, 3), padding = "same",
                 activation = "relu"))

# Fourth convolutional layer
print("Adding fourth Conv2D")
model.add(Conv2D(filters = 256, kernel_size = (3, 3), padding = "same",
                 activation = "relu"))
model.add(MaxPooling2D(pool_size = (1, 2), strides = (2, 2)))

# Fifth convolutional layer
print("Adding fifth Conv2D")
model.add(Conv2D(filters = 512, kernel_size = (3, 3), padding = "same",
                 activation = "relu"))

# First normalization layer
print("Adding first normalization layer")
model.add(BatchNormalization())

# Sixth convolutional layer
print("Adding sixth Conv2D")
model.add(Conv2D(filters = 512, kernel_size = (3, 3), padding = "same",
                 activation = "relu"))

# Second normalization layer
print("Adding second normalization layer")
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size = (1, 2), strides = (2, 2)))

# Seventh convolutional layer
print("Adding seventh Conv2D")
model.add(Conv2D(filters = 512, kernel_size = (2, 2), padding = "valid",
                 activation = "relu"))


# Reshape layer
print("Adding reshape")
model.add(Reshape((4, 1152)))

# LSTM layer
print("Adding bidirectional LSTM layer")
model.add(Bidirectional(LSTM(units = 512, return_sequences = True),
                        merge_mode = "sum"))

# transform RNN output to character activation
# 52 output units: 52 letters -> (upper/lowercase)
print("Adding activation")
model.add(Dense(52, kernel_initializer = "he_normal",
                activation = "softmax"))

model.summary()

# Compile
print("Compiling model")
model.compile(loss="categorical_crossentropy", optimizer="adam",
              metrics=["accuracy"])

# Train
print("Fitting model")
model.fit(X_train, Y_train,
          validation_data = (X_test, Y_test),
          batch_size = 32,
          epochs = 20, verbose = 1)

# Evaluate
print("Evaluating model")
score = model.evaluate(X_test, Y_test, verbose = 1)
print(model.metrics_names)
print(score)

# Save
print("Saving model")
model.save("model.h5")

sys.exit()
