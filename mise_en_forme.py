import argparse
import numpy as np 
import pickle

ap = argparse.ArgumentParser()
ap.add_argument('-apply', '--apply', required = True, type = bool)
args = vars(ap.parse_args())

x_train = np.load(r"/content/x_train.npy")
x_test = np.load(r"/content/x_test.npy")
y_train = np.load(r"/content/y_train.npy")


if args['apply']:
    y_test = np.zeros(len(x_test))
    np.save('/content/y_test.npy', y_test)
else:
    y_test = np.load(r"/content/y_test.npy")


train_label = (([str(i) for i in list(range(len(y_train)))], y_train))
with open('/content/skeleton-based-action-recognition/data_coup/train_label.pkl', 'wb') as handle:
    pickle.dump(train_label, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
test_label = (([str(i) for i in list(range(len(y_test)))], y_test))
with open('/content/skeleton-based-action-recognition/data_coup/test_label.pkl', 'wb') as handle:
    pickle.dump(test_label, handle, protocol=pickle.HIGHEST_PROTOCOL)

x_train = x_train[:, :, :, :, :3]
x_test = x_test[:, :, :, :, :3]

tr1, tr2, tr3, tr4 = x_train.shape[1], x_train.shape[2], x_train.shape[3], x_train.shape[4]
te1, te2, te3, te4 = x_test.shape[1], x_test.shape[2], x_test.shape[3], x_test.shape[4]


x_train = np.reshape(x_train, (len(x_train), tr4, tr2, tr3, tr1))
x_test = np.reshape(x_test, (len(x_test), te4, te2, te3, te1))


np.save('/content/skeleton-based-action-recognition/data_coup/x_train.npy', x_train)
np.save('/content/skeleton-based-action-recognition/data_coup/x_test.npy', x_test)
