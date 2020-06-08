from PIL import Image
import json
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
from predict_result import Predict_Result
import warnings
import logging


class Prediction_Util:
    def __init__(self, image_path, model, top_k =1, category_names = None, image_size=224, pix_range=255, color_channel=3, num_classes =102):
        warnings.filterwarnings('ignore')

        logger = tf.get_logger()
        logger.setLevel(logging.ERROR)
        
        self.image_path = image_path
        self.model = self.__load_model(model)
        self.top_k = top_k
        self.image_size = image_size
        self.pix_range = pix_range
        self.color_channel = color_channel
        self.num_classes = num_classes
        self.category_names = self.__load_category_names(category_names)
        
        
    def process_image(self, image):
        image = tf.convert_to_tensor(image)
        image = tf.cast(image, tf.float32)
        image = tf.image.resize(image , (self.image_size, self.image_size))
        image /= self.pix_range
        return image
    
    def __load_category_names(self, category_names):
        if category_names is None:
            return None
        with open(category_names, 'r') as f:
            return json.load(f)
    
    def get_image_asnparray(self):
        im = Image.open(self.image_path)
        return np.asarray(im)
    
    def __load_model(self, model):
        return tf.keras.models.load_model((model),custom_objects={'KerasLayer':hub.KerasLayer})
    
    def predict(self):
        im = self.get_image_asnparray()
        im = self.process_image(im)
        im = tf.convert_to_tensor(im)
        im = tf.reshape(im, (1, self.image_size, self.image_size, self.color_channel))
        probabilities = self.model.predict(im)
        temp = zip(probabilities.squeeze(), list(range(1,self.num_classes+1)))
        sorted_temp = sorted(temp, key = lambda tmp: tmp[0], reverse=True)
        result= list()
        for prob,label in sorted_temp[:self.top_k]:
            if self.category_names is None:
                result.append(Predict_Result(prob, str(label), None))
            else:
                result.append(Predict_Result(prob, label, self.category_names[str(label)]))
                
        return result
        