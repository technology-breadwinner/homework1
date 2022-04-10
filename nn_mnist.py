# coding=utf-8
import numpy as np
import struct
import os
from tensorboardX import SummaryWriter
from layer import FullyConnectedLayer, ReLULayer, SoftmaxLossLayer


# 文件路径
MNIST_DIR = "D:\\project\\神经网络与深度学习\\mnist"
TRAIN_DATA = "train-images.idx3-ubyte"
TRAIN_LABEL = "train-labels.idx1-ubyte"
TEST_DATA = "t10k-images.idx3-ubyte"
TEST_LABEL = "t10k-labels.idx1-ubyte"

np.random.seed(12)

class MNIST_MLP(object):
    def __init__(self,alpha, batch_size=30, input_size=784, hidden1=128,out_classes=10, lr=0.005, max_epoch=20, print_iter=100):
        self.batch_size = batch_size
        self.input_size = input_size
        self.hidden1 = hidden1
        self.out_classes = out_classes
        self.lr = lr
        self.max_epoch = max_epoch
        self.print_iter = print_iter
        self.alpha = alpha

    def load_mnist(self, file_dir, is_images = 'True'):
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

    def load_data(self):
        print('Loading MNIST data from files...')
        train_images = self.load_mnist(os.path.join(MNIST_DIR, TRAIN_DATA), True)
        train_labels = self.load_mnist(os.path.join(MNIST_DIR, TRAIN_LABEL), False)
        test_images = self.load_mnist(os.path.join(MNIST_DIR, TEST_DATA), True)
        test_labels = self.load_mnist(os.path.join(MNIST_DIR, TEST_LABEL), False)
        self.train_data = np.append(train_images, train_labels, axis=1)
        self.test_data = np.append(test_images, test_labels, axis=1)
        # self.test_data = np.concatenate((self.train_data, self.test_data), axis=0)

    def shuffle_data(self):
        print('Randomly shuffle MNIST data...')
        np.random.shuffle(self.train_data)


    def build_model(self):  # 建立网络结构
        print('Building multi-layer perception model...')
        self.fc1 = FullyConnectedLayer(self.input_size, self.hidden1,self.alpha)
        self.relu1 = ReLULayer()
        self.fc2 = FullyConnectedLayer(self.hidden1, self.out_classes,self.alpha)
        self.softmax = SoftmaxLossLayer()
        self.update_layer_list = [self.fc1, self.fc2]

    def init_model(self):
        print('Initializing parameters of each layer in MLP...')
        for layer in self.update_layer_list:
            layer.init_param()

    def load_model(self, param_dir):
        print('Loading parameters from file ' + param_dir)
        params = np.load(param_dir, allow_pickle=True).item()
        self.fc1.load_param(params['w1'], params['b1'])
        self.fc2.load_param(params['w2'], params['b2'])

    def save_model(self, param_dir):
        print('Saving parameters to file ' + param_dir)
        params = {}
        params['w1'], params['b1'] = self.fc1.save_param()
        params['w2'], params['b2'] = self.fc2.save_param()
        np.save(param_dir, params)

    def forward(self, input):  # 神经网络的前向传播
        h1 = self.fc1.forward(input)
        h1 = self.relu1.forward(h1)
        h2 = self.fc2.forward(h1)
        prob = self.softmax.forward(h2)
        return prob

    def backward(self):  # 神经网络的反向传播
        dloss = self.softmax.backward()
        dh2 = self.fc2.backward(dloss)
        dh1 = self.relu1.backward(dh2)
        dh1 = self.fc1.backward(dh1)

    def update(self, lr):
        for layer in self.update_layer_list:
            layer.update_param(lr)

    def train(self):
        max_batch = self.train_data.shape[0] // self.batch_size
        print('Start training...')
        for idx_epoch in range(self.max_epoch):
            self.shuffle_data()
            for idx_batch in range(max_batch):
                batch_images = self.train_data[idx_batch*self.batch_size:(idx_batch+1)*self.batch_size, :-1]
                batch_labels = self.train_data[idx_batch*self.batch_size:(idx_batch+1)*self.batch_size, -1]
                prob = self.forward(batch_images)
                loss = self.softmax.get_loss(batch_labels)
                self.backward()
                self.update(self.lr)
            print('Epoch %d,loss: %.6f' % (idx_epoch,loss))


    def evaluate(self):
        pred_results = np.zeros([self.test_data.shape[0]])
        for idx in range(self.test_data.shape[0]//self.batch_size):
            batch_images = self.test_data[idx*self.batch_size:(idx+1)*self.batch_size, :-1]
            prob = self.forward(batch_images)
            pred_labels = np.argmax(prob, axis=1)
            pred_results[idx*self.batch_size:(idx+1)*self.batch_size] = pred_labels
        accuracy = np.mean(pred_results == self.test_data[:,-1])
        print('Accuracy in test set: %f' % accuracy)
        return accuracy

def build_mnist_mlp(param_dir='weight.npy'):
    #alpha_scale = [0,0.0001,0.0002,0.0005]
    alpha_scale = [0.0001]
    # lr_scale = [0.001,0.002,0.003,0.004,0.005]
    lr_scale = [0.005]
    # hidden_num = [128,256]
    hidden_num = [256]
    best_lr, best_hidden, best_alpha, best_acc= 0, 0, 0, 0
    for h1 in hidden_num:
        for lr in lr_scale:
            for alpha in alpha_scale:
                mlp = MNIST_MLP(alpha= alpha,hidden1=h1,lr = lr)
                mlp.load_data()
                mlp.build_model()
                mlp.init_model()
                mlp.train()
                test_acc = mlp.evaluate()
                if test_acc > best_acc:
                    best_acc, best_lr, best_hidden, best_alpha= test_acc, lr, h1, alpha
                    mlp.save_model('mlp.npy')
                mlp.save_model('mlp.npy')
    return mlp

if __name__ == '__main__':
    mlp = build_mnist_mlp()
