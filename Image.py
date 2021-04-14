import cv2
import os
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.model_selection import train_test_split
import numpy as np
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Flatten, BatchNormalization
from tensorflow.keras.layers import Dense, MaxPooling2D, Conv2D
from tensorflow.keras.layers import Input, Activation, Add
from tensorflow.keras.models import Model
from tensorflow.keras.regularizers import l2
from tensorflow.keras.optimizers import Adam

i = 0
last = []
images = []
ageList = []
sexList = []


class Image:

    def image_processing(self, folder):
        for filename in os.listdir(folder):
            img = cv2.imread(os.path.join(folder, filename))
            if img is not None:
                labels = self.establishLabels(filename)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img = cv2.resize(img, (48, 48))
                images.append(img)
                sexList.append(labels['sex'])
                ageList.append(labels['age'])

        self.showProcessedImage(23)

        # convert the labels and images to NumPy arraysd

        images_f = np.array(images)
        sexList_f = np.array(sexList)
        ageList_f = np.array(ageList)

        np.save(folder + 'image.npy', images_f)
        np.save(folder + 'gender.npy', sexList_f)
        np.save(folder + 'age.npy', ageList_f)

        self.showSexDistribution(sexList_f)
        self.showAgeDistribution(ageList_f)

        # images_f_2 = images_f / 255
        # sexList_encoded = tf.keras.utils.to_categorical(sexList_f, num_classes=2)
        # X_train, X_test, Y_train, Y_test = train_test_split(images_f_2, sexList_encoded, test_size=0.25)

        #History = self.trainModel(X_train, X_test, Y_train, Y_test)
        #print(History)

        # Pred = Model.predict(self, X_test)
        # print(Pred)



    def establishLabels(self, filename):
        name = filename.split(".")[0]
        splitName = name.split("A")
        s = int(splitName[-1])
        a = int(splitName[-2])
        output = {'sex': s, 'age': a}

        return output

    def showProcessedImage(self, index):

        cv2.imshow("ProcessedImage", images[index])
        print(ageList[index])
        print(sexList[index])

        print("Press any key to exit...")
        cv2.waitKey(0) #waits for any key press

    def showSexDistribution(self, sexList_f):
        values, counts = np.unique(sexList_f, return_counts=True)
        #print(counts) #returns 207, 219

        plt.xlabel("Sex")
        plt.ylabel("Distribution")
        sex = ['Female', 'Male']
        plt.bar(sex, counts)
        plt.show()


    def showAgeDistribution(self, ageList_f):
        values, counts = np.unique(ageList_f, return_counts=True)
        #print(counts)

        plt.plot(values, counts)
        plt.xlabel('ages')
        plt.ylabel('distribution')
        plt.show()

    def Convolution(self, input_tensor, filters):
        x = Conv2D(filters=filters, kernel_size=(3, 3), padding='same', strides=(1, 1), kernel_regularizer=l2(0.001))(
            input_tensor)
        x = Dropout(0.1)(x)
        x = Activation('relu')(x)
        return x


    def model(self, input_shape):
        inputs = Input((input_shape))

        conv_1 = self.Convolution(inputs, 32)
        maxp_1 = MaxPooling2D(pool_size=(2, 2))(conv_1)
        conv_2 = self.Convolution(maxp_1, 64)
        maxp_2 = MaxPooling2D(pool_size=(2, 2))(conv_2)
        conv_3 = self.Convolution(maxp_2, 128)
        maxp_3 = MaxPooling2D(pool_size=(2, 2))(conv_3)
        conv_4 = self.Convolution(maxp_3, 256)
        maxp_4 = MaxPooling2D(pool_size=(2, 2))(conv_4)
        flatten = Flatten()(maxp_4)
        dense_1 = Dense(128, activation='relu')(flatten)
        drop_1 = Dropout(0.2)(dense_1)
        output = Dense(7, activation="sigmoid")(drop_1)
        model = Model(inputs=[inputs], outputs=[output])
        model.compile(loss="categorical_crossentropy", optimizer="Adam",
                      metrics=["accuracy"])
        return model

    def trainModel(self, X_train, X_test, Y_train, Y_test):
        History = Model.fit(X_train, Y_train,
                 batch_size=32,
                 validation_data=(X_test, Y_test),
                 epochs=1000,
                 callbacks=[])
        return History
