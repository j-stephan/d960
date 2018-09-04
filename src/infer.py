import cv2
import numpy as np
import os
import sys

from keras import backend as K
from keras.layers import Activation, Add, BatchNormalization, Bidirectional
from keras.layers import Conv2D, Dense, Input, Lambda, LSTM, MaxPooling2D
from keras.layers import Reshape, TimeDistributed
from keras.models import Model, Sequential
from keras.optimizers import SGD
from keras.utils import plot_model, to_categorical

from shutil import copy

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


# Create mapping from char to int
char_to_int = dict((c, i) for i, c in enumerate(alphabet))
int_to_char = dict((i, c) for i, c in enumerate(alphabet))

def ctc_decode(y_pred):
    result = []
    y_pred_len = np.ones(y_pred.shape[0]) * y_pred.shape[1]
    label = K.get_value(
              K.ctc_decode(y_pred, input_length = y_pred_len, greedy = True)[0][0])[0]
#    print("Label before conversion:")
#    print(label)
    for i in label:
        result.append(int_to_char[i])
    return ''.join(result)

# Input layer
input_data = Input(name="input_data", shape = (img_width, img_height, 1),
                   dtype = "float32")

# First convolutional layer
conv = Conv2D(name = "conv1", filters = 64, kernel_size = (3, 3),
              padding = "same", activation = "relu")(input_data)
conv = MaxPooling2D(name = "max_pool1", pool_size = (2, 2),
                    strides = (2, 2))(conv)

# Second convolutional layer
conv = Conv2D(name = "conv2", filters = 128, kernel_size = (3, 3),
              padding = "same", activation = "relu")(conv)
conv = MaxPooling2D(name = "max_pool2", pool_size = (2, 2),
                    strides = (2, 2))(conv)

# Third convolutional layer
conv = Conv2D(name = "conv3", filters = 256, kernel_size = (3, 3),
              padding = "same", activation = "relu")(conv)

# Fourth convolutional layer
conv = Conv2D(name = "conv4", filters = 256, kernel_size = (3, 3),
              padding = "same", activation = "relu")(conv)
conv = MaxPooling2D(name = "max_pool3", pool_size = (1, 2),
                    strides = (2, 2))(conv)

# Fifth convolutional layer
conv = Conv2D(name = "conv5", filters = 512, kernel_size = (3, 3),
              padding = "same", activation = "relu")(conv)

# First normalization layer
conv = BatchNormalization(name = "batch_norm1")(conv)

# Sixth convolutional layer
conv = Conv2D(name = "conv6", filters = 512, kernel_size = (3, 3),
              padding = "same", activation = "relu")(conv)

# Second normalization layer
conv = BatchNormalization(name = "batch_norm2")(conv)
conv = MaxPooling2D(name = "max_pool4", pool_size = (1, 2),
                    strides = (2, 2))(conv)

# Seventh convolutional layer
conv = Conv2D(name = "conv7", filters = 512, kernel_size = (2, 2),
              padding = "valid", activation = "relu")(conv)
#Model(inputs=input_data, outputs = conv).summary()

# Reshape layer
conv = Reshape((rnn_time_steps, rnn_vec_size), name = "reshape")(conv)

# Bidirectional LSTM layer
lstm = Bidirectional(LSTM(units = 512, return_sequences = True),
                     merge_mode = "sum", name = "lstm")(conv)

# transform RNN output to character activation
prediction = Dense(alphabet_size, kernel_initializer = "he_normal",
                   activation = "softmax", name = "activation")(lstm)

Model(inputs=input_data, outputs = prediction).summary()

model = Model(inputs = [input_data],
              outputs = [prediction])


# Load weights
print("Loading model weights")
model.load_weights("weights.h5", by_name = True)

# Infer
#test_img = cv2.imread("wl2/Tc.jpg")
#test_img = cv2.resize(test_img, (img_width, img_height))
#test_img = cv2.cvtColor(test_img, cv2.COLOR_RGB2GRAY)
#test_img = test_img.astype('float32')
#test_img /= 255
#test_img = test_img.reshape(1, img_width, img_height, 1)

#truth = "Tc"

#pred = model.predict(test_img)
#decoded = ctc_decode(pred)

image_paths = [data_input + "/{0}".format(f)
                for f in os.listdir(data_input)
                    if os.path.isfile(os.path.join(data_input, f))]

# Infer all images
mismatched = 0
img_num = 0
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
    
    # Convert to model format
    grey = grey.reshape(1, img_width, img_height, 1)

    # Remove numbers from name string
    truth = ''.join(i for i in name if not i.isdigit())

    # infer
    pred = model.predict(grey)
    decoded = ctc_decode(pred)

    if decoded != truth:
        print("Mismatch! Decoded: " + decoded + ", actual: " + truth)
        mismatched += 1
        copy(image_path, "mismatched/")
    else:
        copy(image_path, "passed/")

    img_num += 1

print("Mismatch ratio: " + str(mismatched) + "/" + str(img_num))

sys.exit()
