from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout, Flatten, BatchNormalization
from keras.layers import Conv2D
from keras.utils import to_categorical
from keras.callbacks import ModelCheckpoint
from keras import callbacks
from keras.callbacks import EarlyStopping
from keras.layers import MaxPool2D
import os.path 

'''def build_network(self):
    # Smaller 'AlexNet'
    # https://github.com/tflearn/tflearn/blob/master/examples/images/alexnet.py
    print('[+] Building CNN')
    self.network = input_data(shape = [None, SIZE_FACE, SIZE_FACE, 1])
    self.network = conv_2d(self.network, 64, 5, activation = 'relu')
    #self.network = local_response_normalization(self.network)
    self.network = max_pool_2d(self.network, 3, strides = 2)
    self.network = conv_2d(self.network, 64, 5, activation = 'relu')
    self.network = max_pool_2d(self.network, 3, strides = 2)
    self.network = conv_2d(self.network, 128, 4, activation = 'relu')
    self.network = dropout(self.network, 0.3)
    self.network = fully_connected(self.network, 3072, activation = 'relu')
    self.network = fully_connected(self.network, len(EMOTIONS), activation = 'softmax')
    self.network = regression(self.network,
      optimizer = 'momentum',
      loss = 'categorical_crossentropy')
    self.model = tflearn.DNN(
      self.network,
      checkpoint_path = SAVE_DIRECTORY + '/emotion_recognition',
      max_checkpoints = 1,
      tensorboard_verbose = 2
    )
    self.load_model()
'''
def define_model():
    model = Sequential()

    # 1st stage
    model.add(Conv2D(32, 3, input_shape=(48, 48, 1), padding='same',
                    activation='relu'))
    model.add(BatchNormalization())
    model.add(Conv2D(32, 3, padding='same',
                    activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPool2D(pool_size=(2, 2), strides=2))

    # 2nd stage
    model.add(Conv2D(64, 3, padding='same',
                    activation='relu'))
    model.add(BatchNormalization())
    model.add(Conv2D(64, 3, padding='same',
                    activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPool2D(pool_size=(2, 2), strides=2))
    model.add(Dropout(0.25))

    # 3rd stage
    model.add(Conv2D(128, 3, padding='same',
                    activation='relu'))
    model.add(BatchNormalization())
    model.add(Conv2D(128, 3, padding='same',
                    activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPool2D(pool_size=(2, 2), strides=2))
    model.add(Dropout(0.25))

    # FC layers
    model.add(Flatten())
    model.add(Dense(256))
    model.add(Activation("relu"))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))

    model.add(Dense(256))
    model.add(Activation("relu"))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))

    model.add(Dense(256))
    model.add(Activation("relu"))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))

    model.add(Dense(7))
    model.add(Activation('softmax'))


    
    model.compile(loss='categorical_crossentropy',
              optimizer='adam', metrics=['accuracy'])

    return model 

# load model weights
def model_weights(model):
    
    if os.path.exists('models/weights.h5'):
        model.load_weights('models/weights.h5')
    else:
        print('No model to load !')
    return model

