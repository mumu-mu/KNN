import numpy as np
from sklearn import neighbors
import random as random
import time

data = np.genfromtxt('wifi_localization.txt', dtype = 'int')
data  = np.asarray(data)


#to disrupt the data
index = [i for i in range(len(data))]
random.shuffle(index)
data = data[index]

t_label = np.zeros(500)   #predicted label
ave_acc = 0
#to establish training set and testing set
train_features = data[0:1500,0:7]
train_label = data[0:1500,7]
test_features = data[1500:2000,0:7]
test_label = data[1500:2000,7]

for k in range(80):
    s_time = time.time()
    knn = neighbors.KNeighborsClassifier(n_neighbors = k+1)
    knn.fit(train_features,train_label)
    t_label = knn.predict(test_features)
    e_time = time.time()
    acc = knn.score(test_features,test_label) 
    print('k = %d  the accurancy is: %.4f, and it cost %.4f s'  % (k+1, acc, e_time-s_time))
