import argparse
from  prediction_util import Prediction_Util
from predict_result import Predict_Result

import warnings
import logging
import tensorflow as tf



#def main():
    
warnings.filterwarnings('ignore')

logger = tf.get_logger()
logger.setLevel(logging.ERROR)

parser = argparse.ArgumentParser(description='Flower Classifier')
parser.add_argument("image_path", help='Path of the image that needs to be classified')
parser.add_argument("model", help='Model that will be used for classification')
parser.add_argument('--top_k', type=int, default='1',
    help='Return the top KK most likely classes')
parser.add_argument('--category_names', type=str, default=None,
    help='Path to a JSON file mapping labels to flower names')
args = parser.parse_args()

prediction_util = Prediction_Util(args.image_path, args.model, args.top_k, args.category_names)

result = prediction_util.predict()

for r in result:
    print(r)    
    
    
#if __name__ == '__main__':
    #main()