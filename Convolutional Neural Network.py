import imageio
import numpy as np
import pandas as pd
import seaborn as sn
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.metrics import confusion_matrix
from tensorflow.keras.models import Sequential
from tensorflow.python.keras.layers import AveragePooling2D
from tensorflow.python.keras.utils.np_utils import to_categorical
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout

#setez seed-ul pentru a ma asigura ca pentru acelasi model primesc aceleasi rezultate la rulari diferite
tf.random.set_seed(1)

#TRAIN
#citire label-uri imagini de train

train_file = open('./ai-unibuc-24-22-2021/train.txt')
train_labels = np.zeros(30001, 'int')
for i in range(30001):
    train_labels[i] = int(train_file.readline().split(',')[1])
train_file.close()

#citire imagini train
train_imgs = np.zeros((30001, 32, 32), 'int')
for i in range(30001):
    if i < 10:
        train_imgs[i] = imageio.imread(f'./ai-unibuc-24-22-2021/train/00000{i}.png')
    elif 10 <= i < 100:
        train_imgs[i] = imageio.imread(f'./ai-unibuc-24-22-2021/train/0000{i}.png')
    elif 100 <= i < 1000:
        train_imgs[i] = imageio.imread(f'./ai-unibuc-24-22-2021/train/000{i}.png')
    elif 1000 <= i < 10000:
        train_imgs[i] = imageio.imread(f'./ai-unibuc-24-22-2021/train/00{i}.png')
    else:
        train_imgs[i] = imageio.imread(f'./ai-unibuc-24-22-2021/train/0{i}.png')

train_imgs = train_imgs.reshape(30001, 1024)
y_train = to_categorical(train_labels)


#VALIDATION
#citire label-uri imagini de validation
validation_file = open('./ai-unibuc-24-22-2021/validation.txt')
validation_labels = np.zeros(5000, 'int')
for i in range(30001, 35001):
    validation_labels[i - 30001] = int(validation_file.readline().split(',')[1])
validation_file.close()

#citire imagini validation
validation_imgs = np.zeros((5000, 32, 32), 'int')
val_im = []
for i in range(30001, 35001):
    validation_imgs[i - 30001] = imageio.imread(f'./ai-unibuc-24-22-2021/validation/0{i}.png')
    val_im.append(f'0{i}')

y_validation = to_categorical(validation_labels)

#TEST
#citire imagini test
test_imgs = np.zeros((5000, 32, 32), 'int')
test_im = []
for i in range(35001, 40001):
    test_imgs[i - 35001] = imageio.imread(f'./ai-unibuc-24-22-2021/test/0{i}.png')
    test_im.append(f'0{i}.png')

train_imgs = train_imgs.reshape(30001, 1024)
validation_imgs = validation_imgs.reshape(5000, 1024)
test_imgs = test_imgs.reshape(5000, 1024)

#Normalizez datele folosin normalizarea minmax
def normalize_data(train_data, validation_data, test_data):
    scaler = preprocessing.MinMaxScaler()
    scaler.fit(train_data)
    return scaler.transform(train_data), scaler.transform(validation_data), scaler.transform(test_data)

train_imgs, validation_imgs, test_imgs = normalize_data(train_imgs, validation_imgs, test_imgs)

#fac reshape la imagini pentru a le putea transmite ca input layer-ului Conv2D
train_imgs = train_imgs.reshape(30001, 32, 32, 1)
validation_imgs = validation_imgs.reshape(5000, 32, 32, 1)
test_imgs = test_imgs.reshape(5000, 32, 32, 1)

#Creare model
model = Sequential()

#adaugam straturi
model.add(Conv2D(32, kernel_size= 5, activation = 'relu', input_shape = (32, 32, 1)))
model.add(AveragePooling2D())
model.add(Conv2D(128, kernel_size= 5, activation = 'relu'))
model.add(Dropout(rate=.6))
model.add(Flatten())
model.add(Dense(700, activation = "sigmoid"))
model.add(Dropout(rate=.6))
model.add(Dense(9, activation = 'softmax'))

#compilarea modelului
model.compile(optimizer = 'nadam', loss = 'categorical_crossentropy', metrics = ['accuracy'])

#fac rezumatul modelului pentru a vedea cati parametri am
model.summary()

#antrenarea modelului
model_history = model.fit(train_imgs, y_train, validation_data = (validation_imgs, y_validation), batch_size=64, epochs = 15, shuffle=True)

model.evaluate(validation_imgs, y_validation)

#fac predictiile pe datele de test
predictions_test = model.predict(test_imgs)
predictions_test = predictions_test.argmax(axis=-1)

#scriu predictiile facute in fisierul csv
file = pd.DataFrame({'id':test_im, 'label': predictions_test})
file.to_csv('test_predictions_final.csv', index=False)

#matricea de confuzie
validation_predict = model.predict_classes(validation_imgs)
matrice_confuzie = confusion_matrix(validation_labels, validation_predict, labels=None, sample_weight=None, normalize=None)

#plotez matricea de confuzie
df_cm = pd.DataFrame(matrice_confuzie, index=[i for i in range(9)], columns=[i for i in range(9)])
sn.set(font_scale=1.2)  # label size
sn.heatmap(df_cm, annot=True, annot_kws={"size": 10})  # font size
plt.title("Confusion Matrix")
plt.show()

#fac graficul pentru acurateti
plt.plot(model_history.history['accuracy'])
plt.plot(model_history.history['val_accuracy'])
plt.title('Model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper right')
plt.show()

#fac graficul pentru pierderi
plt.plot(model_history.history['loss'])
plt.plot(model_history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper right')
plt.show()
