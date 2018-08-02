import cv2
import keras
import numpy
import os
import sys

from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split

#from keras import backend as K
#K.tensorflow_backend._get_available_gpus()

data_input = "orig"

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


# Input layer
input_data = keras.layers.Input(shape = (64, 64, 1), dtype = 'float32')

# First convolutional layer
print("Adding first Conv2D")
conv = keras.layers.Conv2D(filters = 16,
                           kernel_size = (3, 3),
                           padding = "same",
                           activation = "relu")(input_data)
conv = keras.layers.MaxPooling2D(pool_size = (2, 2), strides = (2, 2))(conv)

# Second convolutional layer
print("Adding second Conv2D")
conv = keras.layers.Conv2D(filters = 16,
                           kernel_size = (3, 3),
                           padding = "same",
                           activation = "relu")(conv)
conv = keras.layers.MaxPooling2D(pool_size = (2, 2), strides = (2, 2))(conv)


# Reshape layer
print("Adding reshape")
conv = keras.layers.Reshape((16, 256))(conv)

# Cuts input size to LSTM
print("Adding dense")
conv = keras.layers.Dense(32, activation="relu")(conv)

# GRU layer #1
print("Adding GRU #1")
gru_1a = keras.layers.GRU(1024, return_sequences = True, kernel_initializer = 'he_normal')(conv)
gru_1b = keras.layers.GRU(1024, return_sequences = True, go_backwards = True, kernel_initializer = 'he_normal')(conv)
gru_1  = keras.layers.Add()([gru_1a, gru_1b])

# GRU layer #2
print("Adding GRU #2")
gru_2a = keras.layers.GRU(1024, return_sequences = True, kernel_initializer = 'he_normal')(gru_1)
gru_2b = keras.layers.GRU(1024, return_sequences = True, go_backwards = True, kernel_initializer = 'he_normal')(gru_1)
gru_2 = keras.layers.Concatenate()([gru_2a, gru_2b])

# transform RNN output to character activation
print("Adding activation")
act = keras.layers.Dense(45, kernel_initializer = "he_normal")(gru_2)
y_pred = keras.layers.Activation("softmax")(act)

# Output layer with 52 nodes (one for each possible letter)
# 45 nodes for testing purposes
#print("Adding output")
#model.add(keras.layers.Dense(45, activation = "softmax"))

model = keras.models.Model(inputs=input_data, outputs=y_pred)
model.summary()

# Compile
print("Compiling model")
model.compile(loss="categorical_crossentropy", optimizer="adam",
              metrics=["accuracy"])

# Train
print("Fitting model")
model.fit(X_train, Y_train,
            validation_data = (X_test, Y_test), batch_size = 32,
                              epochs = 100, verbose = 1)

# Evaluate
print("Evaluating model")
score = model.evaluate(X_test, Y_test, verbose = 1)

sys.exit()
