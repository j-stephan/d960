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
conv = keras.layers.Conv2D(filters = 64,
                           kernel_size = (3, 3),
                           padding = "same",
                           activation = "relu")(input_data)
conv = keras.layers.MaxPooling2D(pool_size = (2, 2), strides = (2, 2))(conv)

# Second convolutional layer
print("Adding second Conv2D")
conv = keras.layers.Conv2D(filters = 128,
                           kernel_size = (3, 3),
                           padding = "same",
                           activation = "relu")(conv)
conv = keras.layers.MaxPooling2D(pool_size = (2, 2), strides = (2, 2))(conv)

# Third convolutional layer
print("Adding third Conv2D")
conv = keras.layers.Conv2D(filters = 256,
                           kernel_size = (3, 3),
                           padding = "same",
                           activation = "relu")(conv)

# Fourth convolutional layer
print("Adding fourth Conv2D")
conv = keras.layers.Conv2D(filters = 256,
                           kernel_size = (3, 3),
                           padding = "same",
                           activation = "relu")(conv)
conv = keras.layers.MaxPooling2D(pool_size = (1, 2), strides = (2, 2))(conv)

# Fifth convolutional layer
print("Adding fifth Conv2D")
conv = keras.layers.Conv2D(filters = 512,
                           kernel_size = (3, 3),
                           padding = "same",
                           activation = "relu")(conv)

# First normalization layer
print("Adding first normalization layer")
conv = keras.layers.BatchNormalization()(conv)

# Sixth convolutional layer
print("Adding sixth Conv2D")
conv = keras.layers.Conv2D(filters = 512,
                           kernel_size = (3, 3),
                           padding = "same",
                           activation = "relu")(conv)

# Second normalization layer
print("Adding second normalization layer")
conv = keras.layers.BatchNormalization()(conv)
conv = keras.layers.MaxPooling2D(pool_size = (1, 2), strides = (2, 2))(conv)

# Seventh convolutional layer
print("Adding seventh Conv2D")
conv = keras.layers.Conv2D(filters = 512,
                           kernel_size = (2, 2),
                           padding = "valid",
                           activation = "relu")(conv)


# Reshape layer
print("Adding reshape")
seq = keras.layers.Reshape((9, 512))(conv)

# Upper LSTM layer
print("Adding bidirectional LSTM layer")
lstm_a = keras.layers.LSTM(units = 512)(seq)

# Lower LSTM layer
lstm_b = keras.layers.LSTM(units = 512, go_backwards = True)(seq)

# Add
result = keras.layers.Add()([lstm_a, lstm_b])

keras.models.Model(inputs = input_data, outputs = result).summary()

# transform RNN output to character activation
print("Adding activation")
act = keras.layers.Dense(45, kernel_initializer = "he_normal")(result)
y_pred = keras.layers.Activation("softmax")(act)

model = keras.models.Model(inputs=input_data, outputs=y_pred)
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
          epochs = 100, verbose = 1)

# Evaluate
print("Evaluating model")
score = model.evaluate(X_test, Y_test, verbose = 1)
print(model.metrics_names)
print(score)

# Save
print("Saving model")
model.save("model.h5")

sys.exit()
