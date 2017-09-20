# Installing Tensorflow
# Install Tensorflow from the website: https://www.tensorflow.org/versions/r0.12/get_started/os_setup.html

# Installing Keras
# pip install --upgrade keras

# Installing Theano
# pip install --upgrade --no-deps git+git://github.com/Theano/Theano.git

# Part 1 - Building the CNN

# Importing the Keras libraries and packages
import numpy as np
from keras.models import Sequential
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.optimizers import SGD


# Initialising the CNN
classifier = Sequential()

# Step 1 - Convolution
classifier.add(Convolution2D(32, 3, 3, input_shape = (64, 64, 3), activation = 'relu'))

# Step 2 - Pooling
classifier.add(MaxPooling2D(pool_size = (2, 2)))

# Adding a second convolutional layer
classifier.add(Convolution2D(32, 3, 3, activation = 'relu'))
classifier.add(MaxPooling2D(pool_size = (2, 2)))

# Step 3 - Flattening
classifier.add(Flatten())

# Step 4 - Full connection
classifier.add(Dense(output_dim = 128, activation = 'relu'))
classifier.add(Dense(output_dim = 1, activation = 'sigmoid'))

# Compiling the CNN
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

# Part 2 - Fitting the CNN to the images

from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img

img = load_img('data/test/vaseline/vaseline001.jpg')  # this is a PIL image
x = img_to_array(img)  # this is a Numpy array with shape (3, 150, 150)
x = x.reshape((1,) + x.shape)  # this is a Numpy array with shape (1, 3, 150, 150)

translearn_datagen = ImageDataGenerator(rescale = 1./255,
                                        shear_range = 0.2,
                                        zoom_range = 0.2,
                                        horizontal_flip = True,
                                        fill_mode = 'nearest')

detect_datagen = ImageDataGenerator(rescale = 1./255,
                                    shear_range = 0.2,
                                    zoom_range = 0.2,
                                    horizontal_flip = True,
                                    fill_mode = 'nearest')


train_datagen = ImageDataGenerator(rescale = 1./255,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True,
                                   fill_mode = 'nearest')

#datagen = ImageDataGenerator(
#        shear_range=0.2,
#        zoom_range=0.2,
#        horizontal_flip=True,
#        fill_mode='nearest')



test_datagen = ImageDataGenerator(rescale = 1./255)

training_set = train_datagen.flow_from_directory('data/train',
                                                 target_size = (64, 64),
                                                 batch_size = 72,
                                                 class_mode = 'binary',
                                                 shuffle = False)
#for batch in train_datagen.flow(x, batch_size=1,
#                          save_to_dir='preview', save_prefix='vaseline', save_format='png'):

vaseline_features_train = classifier.predict_generator(training_set, 450)
# save the output as a Numpy array
np.save(open('vaseline_features_train.npy', 'w'), vaseline_features_train)


test_set = test_datagen.flow_from_directory('data/validation',
                                             target_size = (64, 64),
                                             batch_size = 72,
                                             class_mode = 'binary',
                                             shuffle = False)

vaseline_features_validation = classifier.predict_generator(test_set, 90)
np.save(open('vaseline_features_validation.npy', 'w'), vaseline_features_validation)

classifier.fit_generator(training_set,
                         samples_per_epoch = 72,
                         nb_epoch = 36,
                         steps_per_epoch = 9,
                         nb_val_samples = 72)

output_set = detect_datagen.flow_from_directory('data/test',
                                                target_size = (64, 64),
                                                shuffle = False)

finaloutput_set = detect_datagen.flow_from_directory('data/test',
                                                target_size = (64, 64),
                                                shuffle = False)



vaseline_features_detection = classifier.predict_generator(output_set, 1)
np.save(open('vaseline_features_detection.npy', 'w'), vaseline_features_detection)

classifier.fit_generator(output_set,
                         steps_per_epoch = 1,
                         nb_epoch = 4)
                         

classifier.save_weights('vaseline_1.h5', overwrite=True)

# the .flow() command below generates batches of randomly transformed images
# and saves the results to the `preview/` directory
#i = 0
#for batch in datagen.flow(x, batch_size=1,
#                          save_to_dir='preview', save_prefix='vaseline', save_format='png'):
#    i += 1
#    if i > 20:
#        break  # otherwise the generator would loop indefinitely

# ------------------------transfer learning-------------------------------------

classifier.load_weights('vaseline_1.h5')

sgd = SGD(lr=1e-4, decay=1e-6, momentum=0.9, nesterov=True)

classifier.compile(optimizer = 'SGD', loss = 'binary_crossentropy', metrics = ['accuracy'])



vaseline_features_sgd = classifier.predict_generator(finaloutput_set, 2)
                                                     

#vaseline_features_sgd = classifier.predict_generator(output_set, 1)
np.save(open('vaseline_features_sgd.npy', 'w'), vaseline_features_sgd)



classifier.save_weights('vaseline_2.h5', overwrite=True)
