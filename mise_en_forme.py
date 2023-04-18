import numpy as np 
import pickle


x_train = np.load(r"/content/x_train.npy")
x_test = np.load(r"/content/x_test.npy")
y_train = np.load(r"/content/y_train.npy")
y_test = np.load(r"/content/y_test.npy")


train_label = (([str(i) for i in list(range(len(y_train)))], y_train))
with open('/content/skeleton-based-action-recognition/data_punch/train_label.pkl', 'wb') as handle:
    pickle.dump(train_label, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
test_label = (([str(i) for i in list(range(len(y_test)))], y_test))
with open('/content/skeleton-based-action-recognition/data_punch/test_label.pkl', 'wb') as handle:
    pickle.dump(test_label, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
b, c, d, e = x_train.shape[1], x_train.shape[2], x_train.shape[3], x_train.shape[4]

x_train = np.reshape(x_train, (len(x_train), e, c, d, b))
x_test = np.reshape(x_train, (len(x_test), e, c, d, b))

np.save('/content/skeleton-based-action-recognition/data_punch/x_train.npy', x_train)
np.save('/content/skeleton-based-action-recognition/data_punch/x_test.npy', x_test)
