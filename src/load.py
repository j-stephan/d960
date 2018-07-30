import cv2
import keras
import numpy
import os
import sys

from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split

data_input = "data"

image_paths = [data_input + "/{0}".format(f)
                for f in os.listdir(data_input)
                    if os.path.isfile(os.path.join(data_input, f))]

grey_output = "greyscale"

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

    # Save image
    if not os.path.exists(grey_output):
        os.makedirs(grey_output)

    out_name = grey_output + "/" + name + ".tif"
    cv2.imwrite(out_name, grey)

    label = name[0]
    data.append(grey)
    labels.append(label)

# Create numpy arrays
data = numpy.array(data)
labels = numpy.array(labels)

# Create training and test data
(X_train, X_test, Y_train, Y_test) = train_test_split(data, labels,
                                        test_size=0.25, random_state=0)

# Convert labels into one-hot encodings for Keras
lb = LabelBinarizer().fit(Y_train)
Y_train = lb.transform(Y_train)
Y_test = lb.transform(Y_test)

X_train = X_train.reshape(X_train.shape[0], 64, 64, 1)
X_test = X_test.reshape(X_test.shape[0], 64, 64, 1)

model = keras.models.Sequential()

# First convolutional layer
model.add(keras.layers.Conv2D(filters = 20,
                              kernel_size = (4, 4),
                              padding = "same",
                              input_shape = (64, 64, 1),
                              activation = "relu"))
model.add(keras.layers.MaxPooling2D(pool_size = (2, 2), strides = (2, 2)))

# Second convolutional layer

model.add(keras.layers.Conv2D(filters = 50,
                              kernel_size = (4, 4),
                              padding = "same",
                              activation = "relu"))
model.add(keras.layers.MaxPooling2D(pool_size = (2, 2), strides = (2, 2)))

# Hidden layer with 500 nodes
model.add(keras.layers.Flatten())
model.add(keras.layers.Dense(500, activation = "relu"))

# Output layer with 52 nodes (one for each possible letter)
# 45 nodes for testing purposes
model.add(keras.layers.Dense(45, activation = "softmax"))

# Compile
model.compile(loss="categorical_crossentropy", optimizer="adam",
              metrics=["accuracy"])

# Train
model.fit(X_train, Y_train,
            validation_data = (X_test, Y_test), batch_size = 32,
                              epochs = 100, verbose = 1)

# Evaluate
score = model.evaluate(X_test, Y_test, verbose = 1)

sys.exit()
