import cv2
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import random
import os
from keras.utils.np_utils import to_categorical
from keras.layers import Dense , Flatten,Dropout
from keras.layers.convolutional import Conv2D , MaxPooling2D
from keras.models import Sequential
import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
import pickle

threshold = 0.75
path = 'Resources/myData'
labelFile = 'Resources/labels.csv'
epochs = 10
batch_size = 75
testratio = 0.2
validationratio = 0.2
imagedimension = (32,32,3)


count = 0
Images = []
classNo = []
myList = os.listdir(path)
print('total no. of classes',len(myList))
noofclasses = len(myList)
print("Importing Classes....")
for x in range(0,len(myList)):
    mypiclist = os.listdir(path+'/'+str(count))
    for y in mypiclist:
        curImg = cv2.imread(path+'/'+str(count)+'/'+y)
        Images.append(curImg)
        classNo.append(count)
    print(count,end =' ')
    count +=1

Images = np.array(Images)
classNo = np.array(classNo)

X_train ,X_test , y_train  , y_test = train_test_split(Images , classNo , test_size = testratio,random_state = 43)
X_train , X_validation , y_train , y_validation = train_test_split(X_train , y_train , test_size = validationratio,random_state = 44)

print('data shape')
print('train',end =' ');print(X_train.shape,y_train.shape)
print('validation',end =' ');print(X_validation.shape,y_validation.shape)
print('test',end =' ');print(X_test.shape,y_test.shape)
assert(X_train.shape[0]==y_train.shape[0])
assert(X_validation.shape[0]==y_validation.shape[0])
assert(X_test.shape[0]==y_test.shape[0])
assert(X_train.shape[1:]==imagedimension)
assert(X_validation.shape[1:]==imagedimension)
assert(X_test.shape[1:]==imagedimension)

data = pd.read_csv(labelFile)
print(data.shape)

def Gray(img):
    img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    return img

def equalize(img):
    img = cv2.equalizeHist(img)
    return img

def preprocessing(img):
    img = Gray(img)
    img = equalize(img)
    img = img/255
    return img

X_train = np.array(list(map(preprocessing,X_train)))
X_validation = np.array(list(map(preprocessing,X_validation)))
X_test = np.array(list(map(preprocessing,X_test)))
cv2.imshow('Gray Scale Images',X_train[random.randint(0,len(X_train)-1)])

X_train = X_train.reshape(X_train.shape[0],X_train.shape[1],X_train.shape[2],1)
X_validation = X_validation.reshape(X_validation.shape[0],X_validation.shape[1],X_validation.shape[2],1)
X_test = X_test.reshape(X_test.shape[0],X_test.shape[1],X_test.shape[2],1)

Datagen = ImageDataGenerator(width_shift_range=0.1,   # 0.1 = 10%     IF MORE THAN 1 E.G 10 THEN IT REFFERS TO NO. OF  PIXELS EG 10 PIXELS
                            height_shift_range=0.1,
                            zoom_range=0.2,  # 0.2 MEANS CAN GO FROM 0.8 TO 1.2
                            shear_range=0.1,  # MAGNITUDE OF SHEAR ANGLE
                            rotation_range=10)

Datagen.fit(X_train)
batches = Datagen.flow(X_train,y_train,batch_size=20)

y_train = to_categorical(y_train,noofclasses)
y_validation = to_categorical(y_validation,noofclasses)
y_test = to_categorical(y_test,noofclasses)


def myModel():
    no_of_filters=60
    size_of_filters=(5,5)

    size_of_filters2=(3,3)
    size_of_pool=(2,2)
    No_of_nodes = 500
    model = Sequential()
    model.add((Conv2D(no_of_filters,size_of_filters,input_shape=(imagedimension[0],imagedimension[1],1),activation='relu')))
    model.add((Conv2D(no_of_filters,size_of_filters,activation='relu')))
    model.add((MaxPooling2D(poo     l_size=size_of_pool)))

    model.add((Conv2D(no_of_filters//2,size_of_filters2,activation='relu')))
    model.add((Conv2D(no_of_filters//2,size_of_filters2,activation='relu')))
    model.add((MaxPooling2D(pool_size=size_of_pool)))

    model.add(Flatten())
    model.add(Dense(No_of_nodes,activation='relu'))
    model.add(Dense(noofclasses,activation='softmax'))

    model.compile(optimizer='Adam',loss='categorical_crossentropy',metrics=['accuracy'])
    return model

model = myModel()
history = model.fit_generator(Datagen.flow(X_train,y_train,batch_size=50),steps_per_epoch=250,epochs = epochs,validation_data=(X_validation,y_validation))

plt.figure(1)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.legend(['training','validation'])
plt.title('loss')
plt.xlabel('epoch')
plt.figure(2)
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.legend(['training','validation'])
plt.title('Acurracy')
plt.xlabel('epoch')
plt.show()
model.save('traffic_model')
# score =model.evaluate(X_test,y_test,verbose=0)
# print('Test Score:',score[0])
# print('Test Accuracy:',score[1])

# pickle_out= open("model_trained.p","wb")  # wb = WRITE BYTE
# pickle.dump(model,pickle_out)
# pickle_out.close()
# cv2.waitKey(0)
# frameWidth = 640  # CAMERA RESOLUTION
# frameHeight = 480
# brightness = 180
# threshold = 0.75  # PROBABLITY THRESHOLD
# font = cv2.FONT_HERSHEY_SIMPLEX
# ##############################################
#
# # SETUP THE VIDEO CAMERA
# cap = cv2.VideoCapture(0)
# cap.set(3, frameWidth)
# cap.set(4, frameHeight)
# cap.set(10, brightness)
# cap = cv2.VideoCapture(0)
#
# # pickle_out = open('./model_trained.p','rb')
# # model = pickle.load(pickle_out)
#
# def Gray(img):
#     img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
#     return img
#
# def equalize(img):
#     img = cv2.equalizeHist(img)
#     return img
# def preprocessing(img):
#     img = Gray(img)
#     img = equalize(img)
#     img = img/255
#     return img
#
# def getClassName(classNo):
#     if   classNo == 0: return 'Speed Limit 20 km/h'
#     elif classNo == 1: return 'Speed Limit 30 km/h'
#     elif classNo == 2: return 'Speed Limit 50 km/h'
#     elif classNo == 3: return 'Speed Limit 60 km/h'
#     elif classNo == 4: return 'Speed Limit 70 km/h'
#     elif classNo == 5: return 'Speed Limit 80 km/h'
#     elif classNo == 6: return 'End of Speed Limit 80 km/h'
#     elif classNo == 7: return 'Speed Limit 100 km/h'
#     elif classNo == 8: return 'Speed Limit 120 km/h'
#     elif classNo == 9: return 'No passing'
#     elif classNo == 10: return 'No passing for vechiles over 3.5 metric tons'
#     elif classNo == 11: return 'Right-of-way at the next intersection'
#     elif classNo == 12: return 'Priority road'
#     elif classNo == 13: return 'Yield'
#     elif classNo == 14: return 'Stop'
#     elif classNo == 15: return 'No vechiles'
#     elif classNo == 16: return 'Vechiles over 3.5 metric tons prohibited'
#     elif classNo == 17: return 'No entry'
#     elif classNo == 18: return 'General caution'
#     elif classNo == 19: return 'Dangerous curve to the left'
#     elif classNo == 20: return 'Dangerous curve to the right'
#     elif classNo == 21: return 'Double curve'
#     elif classNo == 22: return 'Bumpy road'
#     elif classNo == 23: return 'Slippery road'
#     elif classNo == 24: return 'Road narrows on the right'
#     elif classNo == 25: return 'Road work'
#     elif classNo == 26: return 'Traffic signals'
#     elif classNo == 27: return 'Pedestrians'
#     elif classNo == 28: return 'Children crossing'
#     elif classNo == 29: return 'Bicycles crossing'
#     elif classNo == 30: return 'Beware of ice/snow'
#     elif classNo == 31: return 'Wild animals crossing'
#     elif classNo == 32: return 'End of all speed and passing limits'
#     elif classNo == 33: return 'Turn right ahead'
#     elif classNo == 34: return 'Turn left ahead'
#     elif classNo == 35: return 'Ahead only'
#     elif classNo == 36: return 'Go straight or right'
#     elif classNo == 37: return 'Go straight or left'
#     elif classNo == 38: return 'Keep right'
#     elif classNo == 39: return 'Keep left'
#     elif classNo == 40: return 'Roundabout mandatory'
#     elif classNo == 41: return 'End of no passing'
#     elif classNo == 42: return 'End of no passing by vechiles over 3.5 metric tons'
#
# while True:
#     success , imgOriginal = cap.read()
#
#     img = np.array(imgOriginal)
#     img = cv2.resize(img,(32,32))
#     img = preprocessing(img)
#     img = img.reshape(1,32,32,1)
#     # cv2.imshow("processed Image",imgOriginal)
#     cv2.putText(imgOriginal , 'Class: ' , (20,35),cv2.FONT_HERSHEY_SIMPLEX,0.75,(0,0,255),2,cv2.LINE_AA)
#     cv2.putText(imgOriginal, 'Probability: ', (20, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2, cv2.LINE_AA)
#
#     prediction = model.predict(img)
#     print(prediction)
#     class_predict = np.argmax(img,axis=1)
#     print(class_predict)
#     cp = model.predict_classess(img)
#     probability_value = np.amax(prediction)
#     if probability_value > threshold:
#         cv2.putText(imgOriginal,str(class_predict) + ' ' + str(getClassName(class_predict)), (50,35) , cv2.FONT_HERSHEY_COMPLEX , 0.75 , (0,255,0),2,cv2.LINE_AA)
#         cv2.putText(imgOriginal, str(probability_value *100) + ' %' + str(prediction), (50, 35),cv2.FONT_HERSHEY_COMPLEX, 0.75, (0, 255, 0), 2, cv2.LINE_AA)
#         cv2.imshow("processed Image", imgOriginal)
#     if cv2.waitKey(0) & 0xFF == ord('q'):
#         break
#
# cv2.destroyAllWindows()