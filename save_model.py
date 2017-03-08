import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='3'
from keras.models import Sequential
from keras.layers.core import Dense, Activation
from keras.models import model_from_json
# from keras.utils.np_utils import to_categorical
from keras.utils.visualize_util import plot
import numpy as np

# fix random seed for reproducibility
seed = 7
np.random.seed(seed)

input_dim = 8
final_output_dim = 1
# load dataset
dataset = np.loadtxt("pima-indians-diabetes.csv", delimiter=",")

# split into inpt (X) and output (Y) variables
X = dataset[:,0:8]
Y = dataset[:,8]

# create model
model = Sequential()
model.add(Dense(output_dim=10,input_dim=input_dim,init='uniform'))
model.add(Activation('linear'))
model.add(Dense(output_dim=9))
model.add(Activation('relu'))
model.add(Dense(output_dim=final_output_dim, init='uniform'))
model.add(Activation('linear'))

# compile model
model.compile(loss='binary_crossentropy',
                optimizer='adam',
                metrics=['accuracy'])


# fit the model
model.fit(X, Y, nb_epoch=50, batch_size=32, verbose=0)

# serialize model to JSON
model_json = model.to_json()
with open("arch.json", "w") as json_file:
    json_file.write(model_json)
model.save_weights("weights.h5")
print("Saved model to disk")
plot(model, to_file='model.png',show_shapes=True)
import gc; gc.collect()
