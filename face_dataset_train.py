import numpy as np 
import cv2
from tensorflow import keras
from tensorflow.keras.layers import Dense , Activation , Flatten , Conv2D , MaxPool2D , Dropout
from keras.callbacks import ModelCheckpoint

# Augmentation is carried out  to increase the accuracy
def augment(img):
  M=cv2.getRotationMatrix2D((25,25),np.random.randint(-10,11),1)
  img=cv2.warpAffine(img,M,(50,50))
  return img

#Creating model
#C32ADM , C64ADM ,C128ADM , C256ADM , Flatten , Dense(128) , Activation ,Dense (1) , Activation (sigmoid)
path="/content/drive/MyDrive/Deep learning/trained numpy files/TRAIN.npy"
dataset=np.load(path,allow_pickle=True)
train_inputs=[]
train_targets=[]
for img,target in dataset[:3500]:
  train_inputs.append(augment(img))
  train_targets.append(target)
train_inputs=np.array(train_inputs) 
train_targets=np.array(train_targets)
print(train_inputs.shape,train_targets.shape)
normalized_train_inputs=train_inputs/255

model=keras.Sequential()
model.add(Conv2D(32,(3,3),padding='same',input_shape=normalized_train_inputs.shape[1:]))
model.add(Activation('relu'))
model.add(Dropout(0.3))
model.add(MaxPool2D((2,2),strides=(2,2)))
model.add(Conv2D(64,(3,3),padding='same'))
model.add(Activation('relu'))
model.add(Dropout(0.3))
model.add(MaxPool2D((2,2),strides=(2,2)))
model.add(Conv2D(128,(3,3),padding='same'))
model.add(Activation('relu'))
model.add(Dropout(0.3))
model.add(MaxPool2D((2,2),strides=(2,2)))
model.add(Conv2D(256,(3,3),padding='same'))
model.add(Activation('relu'))
model.add(Dropout(0.3))
model.add(MaxPool2D((2,2),strides=(2,2)))
model.add(Flatten())
model.add(Dense(128))
model.add(Activation('relu'))
model.add(Dense(1))
model.add(Activation("sigmoid"))

model.compile(loss='binary_crossentropy',optimizer=keras.optimizers.Adam(learning_rate=0.001),metrics=['accuracy'])
filepath="/content/drive/MyDrive/Deep learning/best_of_facemask.hdf5"
checkpoint=ModelCheckpoint(filepath,monitor='val_loss',verbose=1,save_best_only=True,mode='min')
#monitor='val_loss' ---> here we are looking for min validation loss and max accuracy so by giving  val_loss it is checking in trems of val loss
#verbose=1 ---> it will print the testing accuracy , save_best_only will save the best model only 
# mode='min' ----> when we give val_loss for monitor and mode='max' when we give val_accuraccy for monitor
callbacks_list=[checkpoint]
model.fit(normalized_train_inputs,train_targets,validation_split=0.10,batch_size=50,epochs=30,callbacks=callbacks_list,verbose=1)
