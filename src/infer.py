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
max_length = 2
img_width = 6 * 32
img_height = 64
rnn_time_steps = 16
rnn_ignore = 2 # ignore first outputs of RNN because they are garbage
rnn_vec_size = 1056
alphabet_size = len(alphabet) + 1 # 1 extra for blank label
label_ctc_size = 2 * alphabet_size

test_input = "wl2"
validation_input = "wl6_real"
test_results = "test.csv"
validation_results = "validation_wl6_real.csv"


# Create mapping from char to int
char_to_int = dict((c, i) for i, c in enumerate(alphabet))
int_to_char = dict((i, c) for i, c in enumerate(alphabet))

def ctc_decode(y_preds):
    # tf.keras.backend.ctc_decode returns a tuple of lists of one element
    # that contains the info we are interested in. The following lines
    # extract the needed information from the tuple (of lists of...)
    results = []
    y_pred_len = np.ones(y_preds.shape[0]) * y_preds.shape[1]
    decoded, _ = K.ctc_decode(y_preds, input_length = y_pred_len, greedy = True)
    for label in K.get_value(decoded[0]):
        result = []
        for i in label:
            if i == -1:
                continue # skip blanks
            result.append(int_to_char[i]) # map to characters
        results.append(''.join(result)) # create string
    return results

def infer(logfile, image_paths, model):
    # Infer all images
    mismatched = 0
    img_num = 0
    f = open(logfile, 'w')
    print("truth;decoded;mismatch", file = f)
    greys = []
    truths = []
    print("Loading images...")
    for image_path in image_paths:
        name = os.path.basename(image_path)
        name = os.path.splitext(name)[0] # remove ~1~, ~2~ etc.
        name = os.path.splitext(name)[0] # remove extension

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
        truth = ''.join(i for i in name if not i.isdigit())

        # Append
        greys.append(grey)
        truths.append(truth)

    # Convert to numpy array
    greys = np.array(greys)

    # reshape
    greys = greys.reshape(greys.shape[0], img_width, img_height, 1)

    # Infer
    print("Inferring...")
    preds = model.predict(greys)
    decodeds = ctc_decode(preds)

    for d, t in zip(decodeds, truths):
        mismatch_bool = 1
        if d != t:
            mismatched += 1
            mismatch_bool = 0
        print(t + ";" + d + ";" + str(mismatch_bool), file = f)
        img_num += 1

    print("Mismatch ratio: " + str(mismatched) + "/" + str(img_num))

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
test_paths = [test_input + "/{0}".format(f)
              for f in os.listdir(test_input)
                if os.path.isfile(os.path.join(test_input, f))]
validation_paths = [validation_input + "/{0}".format(f)
                    for f in os.listdir(validation_input)
                        if os.path.isfile(os.path.join(validation_input, f))]

#print("===== Testing =====")
#infer(test_results, test_paths, model)

print("===== Validation =====")
infer(validation_results, validation_paths, model)

sys.exit()
