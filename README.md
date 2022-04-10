**神经网络与深度学习hw1**          
**杨劲松21210980085**
## 1 下载运行
下载完后，运行nn_mnist.py训练模型，运行evaluate.py文件得到模型在测试集上的表现。

## 2 网络结构
### layer.py文件
定义了全连接层，relu层，softmax层的初始化、传播、加载以及参数更新

```python 
class FullyConnectedLayer(object):# 全连接层
    def __init__(self, num_input, num_output,alpha):  # 全连接层初始化  
    def init_param(self, std=0.01):  # 参数初始化   
    def forward(self, input):  # 前向传播计算
    def backward(self, top_diff):  # 反向传播的计算
    def update_param(self, lr):  # 参数更新
    def load_param(self, weight, bias):  # 参数加载
    def save_param(self):  # 参数保存
class ReLULayer(object):# relu层
    def __init__(self):    # relu层初始化
    def forward(self, input):  # 前向传播的计算
    def backward(self, top_diff):  # 反向传播的计算
class SoftmaxLossLayer(object):# softmax层
    def __init__(self):    # softmax层初始化
    def forward(self, input):  # 前向传播的计算
    def get_loss(self, label):   # 计算损失
    def backward(self):  # 反向传播的计算
```


### nn_mnist.py文件
搭建了两层全连接层的网络结构，中间一层隐藏层。其中学习率lr、隐藏层神经元个数h、正则化权重alpha为超参数，为选取最合适，采用网格化搜索的方式，在多种组合中选择测试集结果最优的模型保存。
train_loss,test_loss,test_accuracy可视化通过tensorboardX库里的SummaryWriter函数实现。

```python 
class MNIST_MLP(object):
    def __init__(self,alpha, batch_size=30, input_size=784, hidden1=128,out_classes=10, lr=0.005, max_epoch=20, print_iter=100): #初始化网络
    def load_mnist(self, file_dir, is_images = 'True'): #加载mnist
    def load_data(self):   #下载mnist数据
    def shuffle_data(self): #打乱数据
    def build_model(self):  # 建立网络结构
    def init_model(self): #初始化模型
    def load_model(self, param_dir): #加载模型
    def save_model(self, param_dir): #保存模型
    def forward(self, input):  # 神经网络的前向传播
    def backward(self):  # 神经网络的反向传播
    def update(self, lr): #更新参数
    def train(self):  #训练模型
    def evaluate(self):  #测试集评估
```

### evaluate.py文件
加载nn_mnist中训练好的模型。，并对测试集进行评估。






