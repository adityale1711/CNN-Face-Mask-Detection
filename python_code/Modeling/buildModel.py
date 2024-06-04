import os
import keras_tuner as kt

from keras.models import Sequential
from keras.optimizers import Adam, RMSprop, SGD
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau

class TrainModel:
    def __init__(self, model_name):
        self.epochs = 100
        self.batch_size = 256
        self.es_patience = 50
        self.lr_patience = 10
        self.target_size = (35, 35)

        self.model_name = model_name

        self.datagen = ImageDataGenerator(rescale=(1.0 / 255), horizontal_flip=True, zoom_range=0.1, shear_range=0.2,
                                     width_shift_range=0.1,
                                     height_shift_range=0.1, rotation_range=4, vertical_flip=False)
        self.val_datagen = ImageDataGenerator(rescale=(1.0 / 255))
        self.train_generator = self.datagen.flow_from_directory(directory='../../splitted_datasets/train',
                                                                target_size=self.target_size, class_mode='categorical',
                                                                batch_size=self.batch_size, shuffle=True)
        self.val_generator = self.val_datagen.flow_from_directory(directory='../../splitted_datasets/val',
                                                                  target_size=self.target_size, class_mode='categorical',
                                                                  batch_size=self.batch_size, shuffle=True)
        self.test_generator = self.val_datagen.flow_from_directory(directory='../../splitted_datasets/test',
                                                                   target_size=self.target_size, class_mode='categorical',
                                                                   batch_size=self.batch_size, shuffle=True)

    def build_model(self, hp):
        model = Sequential()

        num_conv_layers = hp.Int('num_conv_layers', min_value=1, max_value=5, step=1)

        for i in range(num_conv_layers):
            num_conv_filters = hp.Choice(f'num_conv_filters_{i}', values=[8, 16, 32, 64])
            kernel_size = hp.Choice(f'kernel_size_{i}', values=[3, 5])
            stride = hp.Choice(f'stride_{i}', values=[1, 3])

            if i == 0:
                model.add(Conv2D(num_conv_filters, kernel_size=kernel_size, strides=stride, padding='same',
                                 activation='relu', input_shape=(35, 35, 3)))
                model.add(MaxPooling2D(padding='same'))
            else:
                model.add(Conv2D(num_conv_filters, kernel_size=kernel_size, strides=stride, padding='same',
                                 activation='relu'))
                model.add(MaxPooling2D(padding='same'))

        model.add(Flatten())

        num_hidden_layers = hp.Int('num_hidden_layers', min_value=1, max_value=5, step=1)
        for i in range(num_hidden_layers):
            num_units = hp.Choice(f'num_units_{i}', values=[8, 16, 32, 64])
            dropout_rate = hp.Float(f'dropout_rate_{i}', min_value=0.1, max_value=0.9, step=0.1)

            model.add(Dense(num_units, activation='relu'))
            model.add(Dropout(dropout_rate))

            if (i == (num_hidden_layers - 1)):
                model.add(Dense(2, activation='sigmoid'))

        optimizer = hp.Choice('optimizer', values=['Adam', 'RMSprop', 'SGD'])
        loss_function = hp.Choice('loss_function', values=['binary_crossentropy', 'categorical_crossentropy'])
        learning_rate = hp.Choice('learning_rate', values=[1e-5, 1e-4, 1e-3, 3e-3, 1e-2, 3e-2, 0.1, 0.3])
        if optimizer == 'Adam':
            opt = Adam(learning_rate=learning_rate)
            model.compile(loss=loss_function, optimizer=opt, metrics=['accuracy'])

        if optimizer == 'RMSprop':
            opt = RMSprop(learning_rate=learning_rate)
            model.compile(loss=loss_function, optimizer=opt, metrics=['accuracy'])

        if optimizer == 'SGD':
            opt = SGD(learning_rate=learning_rate)
            model.compile(loss=loss_function, optimizer=opt, metrics=['accuracy'])

        return model

    def train_model(self):
        if not os.path.exists('models'):
            os.makedirs('models')

        model_name = 'models/' + self.model_name + '.h5'
        model_name = rename_file_if_exists(model_name)

        cp = ModelCheckpoint(f'{model_name}', monitor='val_accuracy', save_best_only=True,
                             save_weights_only=False, mode='auto', save_freq='epoch', verbose=1)

        early_stopping = EarlyStopping(monitor='val_accuracy', patience=self.es_patience)
        reduce_lr = ReduceLROnPlateau(monitor='val_accuracy', factor=0.2, patience=self.lr_patience)

        if not os.path.exists('tuners'):
            os.makedirs('tuners')

        hyperband_tuner = kt.Hyperband(
            self.build_model,
            objective='val_accuracy',
            max_epochs=self.epochs,
            factor=3,
            seed=42,
            directory='tuners',
            project_name=self.model_name
        )

        best_hyperband_param = {}
        hyperband_tuner.search(self.train_generator, validation_data=self.val_generator, epochs=self.epochs,
                               batch_size=self.batch_size, shuffle=True)

        hyperband_results = hyperband_tuner.results_summary()
        best_hyperband_param = hyperband_tuner.get_best_hyperparameters(num_trials=1)[0]
        best_hyperband_model = hyperband_tuner.hypermodel.build(best_hyperband_param)

        best_hyperband_model = best_hyperband_model.fit(self.train_generator, validation_data=self.val_generator,
                                                        epochs=self.epochs, batch_size=self.batch_size,
                                                        callbacks=[cp, early_stopping, reduce_lr], shuffle=True)

        return best_hyperband_param, hyperband_results, best_hyperband_model

def rename_file_if_exists(file_path):
    if not os.path.exists(file_path):
        return file_path

    index = 1
    base_name, ext = os.path.splitext(file_path)
    new_file_path = f'{base_name}_{index}{ext}'

    while os.path.exists(new_file_path):
        index += 1
        new_file_path = f'{base_name}_{index}{ext}'

    return new_file_path