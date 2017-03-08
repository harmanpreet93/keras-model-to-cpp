import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='3'
import numpy as np
np.random.seed(1336)
from keras.models import Sequential, model_from_json
import json
import argparse
from keras import backend as K

np.set_printoptions(threshold=np.inf)
parser = argparse.ArgumentParser(description='This is a script to run Keras model from saved architecture and weights.')

parser.add_argument('-a', '--architecture', help="JSON with model architecture", required=True)
parser.add_argument('-w', '--weights', help="Model weights in HDF5 format", required=True)
parser.add_argument('-i', '--test_input', help="Test file to evaluate", required=True)
args = parser.parse_args()

print('Read architecture from', args.architecture)
print('Read weights from', args.weights)
print('Read weights from', args.test_input)

arch = open(args.architecture).read()
model = model_from_json(arch)
model.load_weights(args.weights)
arch = json.loads(arch)

# compile model
model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])

file_path = args.test_input
a_file = open(file_path, encoding='utf-8')
n_features = int(float(a_file.readline()))
X = a_file.readline().split(' ')
X[-1] = X[-1].replace('\n','')
Y = a_file.readline()
X = np.reshape(np.asarray([float(i) for i in X]),(1,n_features))
Y = np.reshape(np.asarray(float(Y)),(1,1))
# print(X,Y)
a_file.close()

# evaluate the model
score = model.evaluate(X, Y, verbose=0)
print('%s: %.2f%%' %(model.metrics_names[1], score[1]*100))
print("Predicted Class: ",model.predict_classes(X)[0][0])
print("Actual Class: ",int(Y[0][0]))
print("Predicted value: ",model.predict(X)[0][0])
import gc; gc.collect()
