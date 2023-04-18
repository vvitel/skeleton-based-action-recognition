import numpy as np 
import os
import pickle

os.mkdir('/content/skeleton-based-action-recognition/data_coup/')

x_train = np.load(r"/content/x_train.npy")
x_test = np.load(r"/content/x_test.npy")
y_train = np.load(r"/content/y_train.npy")
y_test = np.load(r"/content/y_test.npy")


train_label = (([str(i) for i in list(range(len(y_test)))], y_train))
with open('U:/MODELE/ilyes/data_corrigée/test_label.pkl', 'wb') as handle:
    pickle.dump(test_label, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
test_label = (([str(i) for i in list(range(len(y_test)))], y_test))
with open('U:/MODELE/ilyes/data_corrigée/test_label.pkl', 'wb') as handle:
    pickle.dump(test_label, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
    
