

import scipy.io as sio
import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
from keras import layers





matfn = ''
data = sio.loadmat(matfn)
Rxx_input = data['Rxx_input']
Rxx_input = tf.convert_to_tensor(Rxx_input)


matfn = ''
data = sio.loadmat(matfn)
label = data['label']
label = tf.convert_to_tensor(label)







input = layers.Input(shape=(64, 64, 2))
x = layers.Conv2D(64,(3, 3), strides=2, padding="same")(input)
x = layers.ReLU()(x)
x = layers.BatchNormalization()(x)
x = layers.Conv2D(64, (3, 3), strides=2, padding="same")(x)
x = layers.ReLU()(x)
x = layers.BatchNormalization()(x)
x = layers.Conv2D(64, (2, 2), strides=2, padding="same")(x)
x = layers.ReLU()(x)
x = layers.BatchNormalization()(x)
x = layers.Conv2D(64, (2, 2), strides=2,padding="same")(x)
x = layers.ReLU()(x)
x = layers.BatchNormalization()(x)
x = layers.Flatten()(x)
x = layers.Dense(400, activation='relu')(x)
x = layers.Dropout(rate=0.2)(x)
x = layers.Dense(200, activation='relu')(x)
x = layers.Dropout(rate=0.1)(x)
x = layers.Dense(50, activation='softmax')(x) #classification


rlr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss',factor=0.6,patience=5,verbose=1)
cbks = [rlr]
CNN_angle = keras.Model(input, x)
CNN_angle.compile(optimizer=keras.optimizers.Adam(learning_rate=0.0002), loss='categorical_crossentropy',metrics=[tf.keras.metrics.CategoricalAccuracy(name="acc")])
CNN_angle.summary()

train_history = CNN_angle.fit(
    x=Rxx_input,
    y=label,
    epochs = 50,
    batch_size = 32,
    shuffle=True,
    validation_split=0.1,
    callbacks=cbks
)




# summarize history for accuracy
f1 = plt.figure(1)
plt.plot(train_history.history['acc'], label='Training accuracy')
plt.plot(train_history.history['val_acc'], label='Validation accuracy')
#plt.title('model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Val.'], loc='lower right')
plt.grid()
plt.show()

# summarize history for loss
f2 = plt.figure(2)
plt.plot(train_history.history['loss'], label='Training loss')
plt.plot(train_history.history['val_loss'], label='Validation loss')
#plt.title('model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Val.'], loc='upper left')
plt.grid()
plt.show()


# In[17]:


# save the training performance for reporting
f3 = plt.figure(1)
plt.subplot(2,1,1)
plt.plot(train_history.history['acc'], label='Training accuracy')
plt.plot(train_history.history['val_acc'], label='Validation accuracy')
plt.ylabel('Accuracy')
plt.legend(['Train', 'Val.'], loc='lower right')
plt.grid()
plt.subplot(2,1,2)
plt.plot(train_history.history['loss'], label='Training loss')
plt.plot(train_history.history['val_loss'], label='Validation loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Val.'], loc='upper right')
plt.grid()
plt.show()
#保存矢量图
plt.savefig("test.svg", dpi=300,format="svg")

CNN_angle.save( 'CNN_angle1.h5')




import pandas as pd
pd.DataFrame(train_history.history).plot(figsize=(8,5))
plt.grid(True)
plt.gca().set_ylim(0, 0.1)
plt.ylabel("loss")
plt.xlabel("Epoch")
plt.title("the loss changes as training the CNN_theta")
plt.show()

# 存储成 .mat文件 原始的都是list
acc = train_history.history['acc']
val_acc = train_history.history['val_acc']
loss = train_history.history['loss']
val_loss = train_history.history['val_loss']


file_mame = 'acc.mat'
sio.savemat(file_mame,{"acc":acc})

file_mame = 'val_acc.mat'
sio.savemat(file_mame,{"val_acc":val_acc})

file_mame = 'loss.mat'
sio.savemat(file_mame,{"loss":loss})

file_mame = 'val_loss.mat'
sio.savemat(file_mame,{"val_loss":val_loss})

