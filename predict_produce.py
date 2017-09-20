from glob import glob
from keras.models import Sequential, Model
from keras.preprocessing.image import array_to_img, img_to_array, load_img
import sys
from keras.preprocessing import image
import numpy as np
from keras.models import model_from_json

def path_to_tensor(img_path):
    # loads RGB image as PIL.Image.Image type
    img = image.load_img(img_path, target_size=(250, 250))
    # convert PIL.Image.Image type to 3D tensor with shape (250, 250, 3)
    x = image.img_to_array(img)
    # normalize the image
    x /= 255
    # convert 3D tensor to 4D tensor with shape (1, 250, 250, 3) and return 4D tensor
    return np.expand_dims(x, axis=0)

def top_produce_predictor(img_path):
    """Depends on the path to tensor function, having a loaded model, and the fruit
    names array being created"""
    from keras.applications.vgg16 import VGG16
    # convert image file to a 4D tensor with shape (1,250,250,3)
    # tensor has normed pixel values
    normed_array = path_to_tensor(img_path)
    # extract the bottleneck features
    extracted_features = VGG16(weights='imagenet', include_top=False).predict(normed_array)
    # make predictions on the features
    preds = loaded_model.predict(extracted_features)
    return fruit_names[np.argmax(preds)]

fruit_names = [item[11:-1] for item in sorted(glob("data/train/*/"))]
impath = sys.argv[1]
json_file = open('bestmodel.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights("tflearningwclassweights02-weights-improvement-16-0.84.hdf5")
print("Loaded model from disk")

# evaluate loaded model on test data
loaded_model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
print(top_produce_predictor(impath))
