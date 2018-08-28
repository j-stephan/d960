import cv2
import numpy as np
import os
import sys

from keras import backend as K
from keras.layers import Activation, Add, BatchNormalization, Bidirectional
from keras.layers import Conv2D, Dense, Embedding, Flatten, Input, Lambda, LSTM, MaxPooling2D
from keras.layers import Reshape, TimeDistributed
from keras.models import Model, Sequential
from keras.optimizers import SGD
from keras.utils import to_categorical

alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
data_input = "wl2"
max_length = 2
img_width = max_length * 64
img_height = 64
rnn_time_steps = 16
rnn_ignore = 2 # ignore first outputs of RNN because they are garbage
rnn_vec_size = 672
#rnn_vec_size = 288
alphabet_size = len(alphabet) + 1 # 1 extra for blank label
label_ctc_size = 2 * alphabet_size

image_paths = [data_input + "/{0}".format(f)
                for f in os.listdir(data_input)
                    if os.path.isfile(os.path.join(data_input, f))]

data = []
labels = []

# Shamelessly stolen from Keras' image_ocr.py example
def ctc_lambda_func(args):
    y_pred, labels, input_length, label_length = args
    y_pred = y_pred[:, rnn_ignore:, :]
    return K.ctc_batch_cost(labels, y_pred, input_length, label_length)

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
    image = cv2.resize(image, (img_width, img_height))

    # Convert to greyscale
    grey = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    # Convert to float and normalize
    grey = grey.astype('float32')
    grey /= 255

    # Remove numbers from name string
    label = ''.join(i for i in name if not i.isdigit())
    data.append(grey)
    labels.append(label)

# Create mapping from char to int
char_to_int = dict((c, i) for i, c in enumerate(alphabet))
int_to_char = dict((i, c) for i, c in enumerate(alphabet))

# Convert labels into integers for Keras
labels[:] = [[char_to_int[c] for c in lb] for lb in labels]
labels_onehot = to_categorical(labels, alphabet_size)
labels_merged = []
# FIXME: Expand to wl > 2
for arr in labels_onehot:
    labels_merged.append(arr[0] + arr[1])

# Create numpy arrays
X = np.array(data)
Y = np.array(labels)
Y_onehot = np.array(labels_onehot)
Y_merged = np.array(labels_merged)

X = X.reshape(X.shape[0], img_width, img_height, 1)

# Input layer
input_data = Input(name="input_data", shape = (img_width, img_height, 1),
                   dtype = "float32")

# First convolutional layer
conv = Conv2D(filters = 64, kernel_size = (3, 3), padding = "same",
              activation = "relu")(input_data)
conv = MaxPooling2D(pool_size = (2, 2), strides = (2, 2))(conv)

# Second convolutional layer
conv = Conv2D(filters = 128, kernel_size = (3, 3), padding = "same",
              activation = "relu")(conv)
conv = MaxPooling2D(pool_size = (2, 2), strides = (2, 2))(conv)

# Third convolutional layer
conv = Conv2D(filters = 256, kernel_size = (3, 3), padding = "same",
              activation = "relu")(conv)

# Fourth convolutional layer
conv = Conv2D(filters = 256, kernel_size = (3, 3), padding = "same",
              activation = "relu")(conv)
conv = MaxPooling2D(pool_size = (1, 2), strides = (2, 2))(conv)

# Fifth convolutional layer
conv = Conv2D(filters = 512, kernel_size = (3, 3), padding = "same",
              activation = "relu")(conv)

# First normalization layer
conv = BatchNormalization()(conv)

# Sixth convolutional layer
conv = Conv2D(filters = 512, kernel_size = (3, 3), padding = "same",
              activation = "relu")(conv)

# Second normalization layer
conv = BatchNormalization()(conv)
conv = MaxPooling2D(pool_size = (1, 2), strides = (2, 2))(conv)

# Seventh convolutional layer
conv = Conv2D(filters = 512, kernel_size = (2, 2), padding = "valid",
              activation = "relu")(conv)
#Model(inputs=input_data, outputs = conv).summary()

# Reshape layer
conv = Reshape((rnn_time_steps, rnn_vec_size))(conv)

# Bidirectional LSTM layer
lstm = Bidirectional(LSTM(units = 512, return_sequences = True),
                     merge_mode = "sum")(conv)

# transform RNN output to character activation
prediction = Dense(alphabet_size, kernel_initializer = "he_normal",
                   activation = "softmax")(lstm)

Model(inputs=input_data, outputs = prediction).summary()

# CTC loss
labels = Input(name = "labels", shape = [alphabet_size], dtype = "float32")
input_length = Input(name = "input_length", shape = [1], dtype = "int64")
label_length = Input(name = "label_length", shape = [1], dtype = "int64")
loss_out = Lambda(ctc_lambda_func, output_shape = (1,), name = "CTC")(
                  [prediction, labels, input_length, label_length])

model = Model(inputs = [input_data, labels, input_length, label_length],
              outputs = loss_out)

# Compile
model.compile(loss={"CTC": lambda y_true, y_pred: y_pred},
              optimizer="adam", metrics=["accuracy"])
#model = Model(inputs=input_data, outputs = prediction)
#model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
#model.fit(x = X, y = Y_onehot, validation_split = 0.25, batch_size = 32, epochs = 20, verbose = 1)

# Train
input_length_arr = [(rnn_time_steps - rnn_ignore) for x in range(0, X.shape[0])]
il_arr = np.array(input_length_arr)

#label_length_arr = [max_length for y in range(0, Y.shape[0])]
label_length_arr = [label_ctc_size for y in range(0, Y.shape[0])]
ll_arr = np.array(label_length_arr)

model.fit(x = [X, Y_merged, il_arr, ll_arr], y = Y_merged,
          validation_split = 0.25,
          batch_size = 32,
          epochs = 100, verbose = 1)

# Save
print("Saving model")
model.save("model.h5")

sys.exit()
