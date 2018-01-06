from keras.callbacks import CSVLogger, ModelCheckpoint, EarlyStopping
from keras.callbacks import ReduceLROnPlateau
from keras.preprocessing.image import ImageDataGenerator
from cnn import *
from datasets import *

# training parameters
epochs = 10000
base_path = '../trained_models/emotion_models/'

# data generator
data_generator = ImageDataGenerator( featurewise_center=False, featurewise_std_normalization=False, 
									rotation_range=10, width_shift_range=0.1, height_shift_range=0.1, 
									zoom_range=.1, horizontal_flip=True )

# model parameters/compilation
model = CNN_trainer((64, 64, 1), 7) # start training
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()
print('Training dataset:', 'fer2013')

# callbacks
log_file = base_path + 'fer2013' + '_emotion_training.log'

csv_logger = CSVLogger(log_file, append=False) # load model 
early_stop = EarlyStopping('val_loss', patience=50)
reduce_lr = ReduceLROnPlateau('val_loss', factor=0.1, patience=int(50/4), verbose=1)
trained_models_path = base_path + 'fer2013' + '_CNN_training'
model_names = trained_models_path + '.{epoch:02d}-{val_acc:.2f}.hdf5'
model_checkpoint = ModelCheckpoint(model_names, 'val_loss', verbose=1,  save_best_only=True)
callbacks = [model_checkpoint, csv_logger, early_stop, reduce_lr]

# loading dataset
inp_shape = (64, 64, 1)
data_loader = DataManager('fer2013', image_size=inp_shape[:2])
faces, emotions = data_loader._load_fer2013()
faces = preprocess_input(faces)
num_samples, classes = emotions.shape
train_data, val_data = split_data(faces, emotions, 0.2)
train_faces, train_emotions = train_data

model.fit_generator(data_generator.flow(train_faces, train_emotions, 32),steps_per_epoch=len(train_faces) / 32,
                        epochs=epochs, verbose=1, callbacks=callbacks, validation_data=val_data)


