import cv2
import keras
import numpy
import os
import sys

from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split

data_input = "orig"

image_paths = [data_input + "/{0}".format(f)
                for f in os.listdir(data_input)
                    if os.path.isfile(os.path.join(data_input, f))]

data = []
labels = []

# Load model
model = keras.models.load_model("model.h5")

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

    data.append(grey)
    labels.append(name[0])

# get unique labels
u_labels = set(labels)
u_labels = list(u_labels)
u_labels = sorted(u_labels)
print(u_labels)

# Create numpy arrays
data = numpy.array(data)
data = data.reshape(data.shape[0], 64, 64, 1)

# Infer
results = model.predict(data, verbose = 1)

for r, l in zip(results, labels):
    # get index of maximum value
    i = numpy.argmax(r)
    if l != u_labels[i]:
        print("Expected: " + str(l) + ", Result: " + str(u_labels[i]))

sys.exit()
