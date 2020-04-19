import argparse
import matplotlib.pyplot as plt
import numpy as np
import json
import PIL
from PIL import Image
import tensorflow as tf
import tensorflow_datasets as tfds
import tensorflow_hub as hub

def normalize(image):
    image = tf.cast(image, tf.float32)
    image /= 255
    return image

def process_image(image):
    processed_image = tf.convert_to_tensor(image)
    processed_image = tf.image.resize(processed_image, (224, 224))
    processed_image = normalize(processed_image)
    return processed_image.numpy()

def predict(image_path, model, top_k):
    image = Image.open(image_path)
    image = np.asarray(image)
    image = process_image(image)
    image = np.expand_dims(image, axis=0)

    predictions = model.predict(image)
    predicted_classes = []
    probabilities = []

    for i in range(top_k):
        index_of_class = np.argmax(predictions, axis=1)[0]
        probabilities.append(predictions[0, index_of_class])
        predicted_classes.append(index_of_class+1)
        predictions[0, index_of_class] = 0

    return probabilities, predicted_classes

def main():
    print('Using:')
    print('\t\u2022 TensorFlow version:', tf.__version__)
    print('\t\u2022 tf.keras version:', tf.keras.__version__)

    parser = argparse.ArgumentParser(description='Image Classifier')

    parser.add_argument('image_path', help= 'Give the image path', default = '../test_images/hard-leaved_pocket_orchid.jpg', action = 'store', type = str)
    parser.add_argument('model', default = '../model.h5', help = 'Give path to the model', action = 'store', type = str)
    parser.add_argument('--top_k', help = 'Return the top K most likely classes', action = 'store', type = int, default = 1)
    parser.add_argument('--category_names', help = 'Path to a JSON file mapping labels to flower names', default = '../label_map.json', action = 'store')

    args = parser.parse_args()

    model = tf.keras.models.load_model(args.model, custom_objects={'KerasLayer':hub.KerasLayer})

    image_path = args.image_path
    im = Image.open(image_path)
    image = np.asarray(im)
    image = process_image(image)
    image = np.expand_dims(image,axis=0)

    probs, classes = predict(image_path, model, 5)
    print(probs)
    print(classes)

if __name__== "__main__" :
    main()
