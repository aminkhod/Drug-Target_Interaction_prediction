import pandas as pd
from keras.layers import Input, Dense
from keras.models import Model
from keras import optimizers
from keras.layers import Conv1D, MaxPooling2D
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.metrics import *
import numpy as np
import tensorflow as tf
mis=[]
data = pd.read_csv("pima.csv", na_values=mis)
print((data.isna().sum()))
# data.dropna(inplace= True)
inp = 8
X = data.values[:,0:inp]
y = data.values[: , inp]
y_edit= np.array([1 if yinstance==1 else 0 for yinstance in y ])
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

 # 32 floats -> compression of factor 24.5, assuming the input is 784 floats

#####Without prepocessing#########
from sklearn.linear_model import LogisticRegression
logreg = LogisticRegression()
LogregModel = logreg.fit(X_train, y_train)
predicts=LogregModel.predict(X_test)

fpr, tpr, thresholds = roc_curve(y_test,predicts)
logauc= auc(fpr, tpr)
print("AUC without preprocessing:",logauc)

# this is the size of our encoded representations
dim = 4
####PCA#########
pca_model = PCA(n_components=dim )
X_train_new = pca_model.fit_transform(X_train)
X_test_new = pca_model.fit_transform(X_test)

from sklearn.linear_model import LogisticRegression
logreg = LogisticRegression()
LogregModel = logreg.fit(X_train_new, y_train)
predicts=LogregModel.predict(X_test_new )

fpr, tpr, thresholds = roc_curve(y_test,predicts)
log_PCauc= auc(fpr, tpr)
print("AUC with PCA:",log_PCauc)

#######Autoencoder############

# this is our input placeholder
input_img = Input(shape=(inp,))
# "encoded" is the encoded representation of the input
encoded = Dense(inp, activation='relu')(input_img)
encoded = Dense(5, activation='relu')(encoded)
encoded = Dense(dim, activation='relu')(encoded)

decoded = Dense(5, activation='relu')(encoded)
decoded = Dense(inp, activation='relu')(decoded)

# this model maps an input to its reconstruction
autoencoder = Model(input_img, decoded)
# this model maps an input to its encoded representation
encoder= Model(input_img, encoded)


autoencoder.compile(optimizer=optimizers.Adadelta(lr=0.004), loss='mse')
# optimizer=Adadelta(lr=0.004)
# Adam(lr=0.02)

autoencoder.fit(x = X_train, y = X_train,
                epochs=5000,batch_size= 750,
                validation_data=(X_test, X_test))
weight1 = autoencoder.layers[1].get_weights()
weight2 = autoencoder.layers[2].get_weights()
weight3 = autoencoder.layers[3].get_weights()
# weight0 = autoencoder.layers[0].get_weights()
# weight0 = autoencoder.layers[0].get_weights()


i1 = tf.matmul( tf.cast(X_train, tf.float32),weight1[0])+weight1[1]
i2 = tf.matmul(i1,weight2[0])+weight2[1]
i3 = tf.matmul(i2, weight3[0])+weight3[1]


j1 = tf.matmul( tf.cast(X_test, tf.float32),weight1[0])+weight1[1]
j2 = tf.matmul(j1,weight2[0])+weight2[1]
j3 = tf.matmul(j2, weight3[0])+weight3[1]

with tf.Session() as ses:
    encoded_train = ses.run(i3)
with tf.Session() as ses1:
    encoded_test = ses1.run(j3)
# encoded_test = encoder.predict(X_test)
# encoded_train =encoder.predict(X_train)

from sklearn.linear_model import LogisticRegression
logreg = LogisticRegression(solver='lbfgs')
LogregModel = logreg.fit(encoded_train, y_train)
predicts=LogregModel.predict(encoded_test)

fpr, tpr, thresholds = roc_curve(y_test,predicts)
log_encodauc= auc(fpr, tpr)
print("AUC with Autoencoder:",log_encodauc)

1+1