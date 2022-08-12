import tensorflow as tf
import fastai
from keras.callbacks import ReduceLROnPlateau
import numpy as np
import keras
import matplotlib.pyplot as plt
from tqdm.autonotebook import tqdm
from IPython import display
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import zipfile
import os
import shutil
import matplotlib.image
from keras_preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from matplotlib import pyplot
import librosa
from pydub import AudioSegment
import librosa.display
import pickle

training = './ReadText/PD'

data, sampling_rate = librosa.load('./ReadText/HC/ID00_hc_0_0_0.wav')

place = "/ReadText/train"

shows = '/ReadText/testing'


cmap = plt.get_cmap('inferno')
plt.figure(figsize=(8, 8))
healthy = "HC PD".split()
print("./ReadText/")
for s in healthy:
    for filename in os.listdir(f'./ReadText/{s}'):
        if filename == ".ipynb_checkpoints":
            pass
        else:
            if np.random.rand(1) < 0.2:
                print(filename)
                loads = f'./ReadText/{s}/{filename}'
                if filename == ".ipynb_checkpoints":
                    pass
                else:
                    y, sr = librosa.load(
                        loads, mono=True, offset=30, duration=50)
                    x, sq = librosa.load(
                        loads, mono=True, offset=10, duration=50)
                    plt.specgram(y, NFFT=2048, Fs=2, Fc=0, noverlap=128,
                                 cmap=cmap, sides='default', mode='default', scale='dB')
                    plt.axis('off')
                    for i in list("hello"):
                        x, sq = librosa.load(loads, mono=True, offset=int(
                            np.random.rand(1) * 100), duration=50)
                        plt.specgram(x, NFFT=2048, Fs=2, Fc=0, noverlap=128,
                                     cmap=cmap, sides='default', mode='default', scale='dB')
                        plt.axis('off')
                        if list(filename)[5] == "p":
                            print(filename)
                            plt.savefig(
                                f'./ReadText/testing/parkinson/{filename}.png')
                        else:
                            plt.savefig(
                                f'./ReadText/testing/healthy/{filename}.png')
                    for i in list("hello"):
                        x, sq = librosa.load(loads, mono=True, offset=int(
                            np.random.rand(1) * 100), duration=50)
                        plt.specgram(x, NFFT=2048, Fs=2, Fc=0, noverlap=128,
                                     cmap=cmap, sides='default', mode='default', scale='dB')
                        plt.axis('off')
                        if list(filename)[5] == "p":
                            print(filename)
                            plt.savefig(
                                f'./ReadText/testing/parkinson/{filename}ea.png')
                        else:
                            plt.savefig(
                                f'./ReadText/testing/healthy/{filename}ea.png')
                    x, sq = librosa.load(
                        loads, mono=True, offset=20, duration=50)
                    plt.specgram(x, NFFT=2048, Fs=2, Fc=0, noverlap=128,
                                 cmap=cmap, sides='default', mode='default', scale='dB')
                    plt.axis('off')
                    if list(filename)[5] == "p":
                        print(filename)
                        plt.savefig(
                            f'./ReadText/testing/parkinson/{filename}eas.png')
                    else:
                        plt.savefig(
                            f'./ReadText/testing/healthy/{filename}eas.png')
                    plt.clf()
            else:
                if filename == ".ipynb_checkpoints":
                    pass
                else:
                    loads = f'./ReadText/{s}/{filename}'
                    y, sr = librosa.load(
                        loads, mono=True, offset=30, duration=50)
                    x, sq = librosa.load(
                        loads, mono=True, offset=10, duration=50)
                    plt.specgram(y, NFFT=2048, Fs=2, Fc=0, noverlap=128,
                                 cmap=cmap, sides='default', mode='default', scale='dB')
                    plt.axis('off')
                    for i in list("hello"):
                        x, sq = librosa.load(loads, mono=True, offset=int(
                            np.random.rand(1) * 100), duration=50)
                        if list(filename)[5] == "h":
                            print(filename)
                            plt.savefig(
                                f'./ReadText/train/healthy/{filename}.png')
                        else:
                            plt.savefig(
                                f'./ReadText/train/parkinson/{filename}.png')
                    plt.specgram(x, NFFT=2048, Fs=2, Fc=0, noverlap=128,
                                 cmap=cmap, sides='default', mode='default', scale='dB')
                    plt.axis('off')
                    for i in list("hello"):
                        x, sq = librosa.load(loads, mono=True, offset=int(
                            np.random.rand(1) * 100), duration=50)
                        if list(filename)[5] == "p":
                            print(filename)
                            plt.savefig(
                                f'./ReadText/train/parkinson/{filename}ea.png')
                        else:

                            x, sq = librosa.load(
                                loads, mono=True, offset=20, duration=50)
                            plt.specgram(x, NFFT=2048, Fs=2, Fc=0, noverlap=128,
                                         cmap=cmap, sides='default', mode='default', scale='dB')
                            plt.axis('off')
                    if list(filename)[5] == "p":
                        print(filename)
                        plt.savefig(
                            f'./ReadText/train/parkinson/{filename}eas.png')
                    else:
                        plt.savefig(
                            f'./ReadText/train/healthy/{filename}eas.png')
                    plt.clf()


train_datagen = ImageDataGenerator(
    rescale=1./255,
    # width_shift_range=0.2,
    # height_shift_range=0.2,
)

test_datagen = ImageDataGenerator(
    rescale=1./255,
)


train_generator = train_datagen.flow_from_directory(
    place,
    target_size=(128, 128),
    batch_size=12,
    shuffle=True,
    class_mode='binary')
test_generator = test_datagen.flow_from_directory(
    shows,
    target_size=(128, 128),
    batch_size=2,
    shuffle=True,
    class_mode='binary')


# Use conv2d layers and MaxPooling2d to train model
model = tf.keras.models.Sequential([tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 3)),
                                    tf.keras.layers.MaxPooling2D(2, 2),
                                    tf.keras.layers.Conv2D(
                                        2, (3, 3), activation='relu'),
                                    tf.keras.layers.MaxPooling2D(2, 2),
                                    tf.keras.layers.Conv2D(
                                        4, (3, 3), activation='relu'),
                                    tf.keras.layers.MaxPooling2D(2, 2),
                                    tf.keras.layers.Conv2D(
                                        1, (3, 3), activation='relu'),
                                    tf.keras.layers.MaxPooling2D(2, 2),
                                    tf.keras.layers.Dropout(0.5),
                                    tf.keras.layers.Flatten(),
                                    tf.keras.layers.Dense(
                                        2, activation='relu'),
                                    tf.keras.layers.Dense(1, activation='sigmoid')])

metric = ReduceLROnPlateau(monitor="val_loss", patience=5)

model.compile(optimizer=tf.keras.optimizers.Adadelta(
    learning_rate=0.0001), loss="binary_crossentropy", metrics=['accuracy'])
model.summary()


history = model.fit(
    train_generator,
    steps_per_epoch=5,  # 144 images = batch_size * steps
    epochs=10,
    validation_data=test_generator,
    validation_steps=6,  # 60 images = batch_size * steps
    verbose=1,
    callbacks=metric)
