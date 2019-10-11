import numpy as np
from sklearn import neighbors
import random as random
import time

#to download the dataset
data = np.genfromtxt('wifi_localization.txt', dtype = 'int')
data  = np.asarray(data)
knn = neighbors.KNeighborsClassifier(n_neighbors = 3)

#to disrupt the data
index = [i for i in range(len(data))]
random.shuffle(index)
data = data[index]

t_label = np.zeros(500)   #predicted label
#to establish training set and testing set
train_features = data[0:1500,0:7]
train_label = data[0:1500,7]
test_features = data[1500:2000,0:7]
test_label = data[1500:2000,7]


s_time = time.time()
knn.fit(train_features,train_label)
t_label = knn.predict(test_features)
e_time = time.time()
acc = knn.score(test_features,test_label) 
for i in range(len(t_label)):
    print('[%d]  the predicted label is: %d the real label is  %d'
          % (i+1 , t_label[i] , test_label[i]))
print('The accurancy is: %.4f, and it cost %.4f s'  % (acc , e_time-s_time))
