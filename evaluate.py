import numpy as np
import struct
import os
from nn_mnist import MNIST_MLP

MNIST_DIR = "D:\\project\\神经网络与深度学习\\mnist"
TRAIN_DATA = "train-images.idx3-ubyte"
TRAIN_LABEL = "train-labels.idx1-ubyte"
TEST_DATA = "t10k-images.idx3-ubyte"
TEST_LABEL = "t10k-labels.idx1-ubyte"

def load_mnist(file_dir, is_images='True'):
    # Read binary data
    bin_file = open(file_dir, 'rb')
    bin_data = bin_file.read()
    bin_file.close()
    # Analysis file header
    if is_images:
        # Read images
        fmt_header = '>iiii'
        magic, num_images, num_rows, num_cols = struct.unpack_from(fmt_header, bin_data, 0)
    else:
        # Read labels
        fmt_header = '>ii'
        magic, num_images = struct.unpack_from(fmt_header, bin_data, 0)
        num_rows, num_cols = 1, 1
    data_size = num_images * num_rows * num_cols
    mat_data = struct.unpack_from('>' + str(data_size) + 'B', bin_data, struct.calcsize(fmt_header))
    mat_data = np.reshape(mat_data, [num_images, num_rows * num_cols])
    print('Load images from %s, number: %d, data shape: %s' % (file_dir, num_images, str(mat_data.shape)))
    return mat_data

def load_data():
    print('Loading MNIST data from files...')
    train_images = load_mnist(os.path.join(MNIST_DIR, TRAIN_DATA), True)
    train_labels = load_mnist(os.path.join(MNIST_DIR, TRAIN_LABEL), False)
    test_images = load_mnist(os.path.join(MNIST_DIR, TEST_DATA), True)
    test_labels = load_mnist(os.path.join(MNIST_DIR, TEST_LABEL), False)
    train_data = np.append(train_images, train_labels, axis=1)
    test_data = np.append(test_images, test_labels, axis=1)
    return train_data, test_data

train_data, test_data = load_data()

def evaluate():
    net = MNIST_MLP(alpha= 0.0001,hidden1=256,lr = 0.0005)
    net.build_model()
    net.init_model()
    net.load_model('mlp.npy')
    y_prob = net.forward(test_data[:, :-1])
    y_pred = np.argmax(y_prob, axis=1)
    test_acc = np.mean(y_pred == test_data[:, -1])
    print('The test accuracy is ', test_acc)

if __name__ == '__main__':
    evaluate()