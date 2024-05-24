### CNN调整参数实验报告

本报告将从**运行时间**、**Average loss**和**Accuracy**三个方面，展示通过调整模型的**网络层数**、**激活函数**、**学习率、Loss计算、添加残差模块**等所得到的不同的运行结果。

比较的方式是每次仅改变一种参数，其余参数设置为默认值,如下表所示：

|**参数名**|**参数值**|
|:--:|:--:|
|batch-size|64|
|test-batch-size|1000|
|epochs|14|
|Lr|1.0|
|Lr step gamma|0.7|
|no-cuda|False|
|kernel\_size|3\*3|
|stride|1|
|dropout|0.25/0.5|
|Conv2d|3|
|Linear|2|
|隐藏层激活函数|Relu|
|输出层激活函数|Softmax|

#### 学习率相关

- Lr

|**Lr**|**Total time (sec.)**|**Average loss**|**Accuracy**|
|:--:|:--:|:--:|:--:|
|1.0|431.2082631587982|0.0228|9928/10000 (99%)|
|0.8|417.0574035644531|0.0221|9932/10000 (99%)|
|0.6|411.1174216270447|0.0221|9930/10000 (99%)|
|2|412.0057752132416|0.0245|9938/10000 (99%)|
|10|413.073853969574|0.0467|9863/10000 (99%)|

- Lr step gamma

|**Lr step gamma**|**Total time (sec.)**|**Average loss**|**Accuracy**|
|:--:|:--:|:--:|:--:|
|0.7|431.2082631587982|0.0228|9928/10000 (99%)|
|0.9|430.93360114097595|0.0307|9929/10000 (99%)|
|0.5|426.4811236858368|0.0246|9909/10000 (99%)|
|0.2|430.9373998641968|0.0285|9898/10000 (99%)|

#### 激活函数选择

- 隐藏层

|**激活函数**|**Total time (sec.)**|**Average loss**|**Accuracy**|
|:--:|:--:|:--:|:--:|
|Relu|431.2082631587982|0.0228|9928/10000 (99%)|
|Sigmoid|411.9199492931366|0.1492|9542/10000 (95%)|
|Tanh|413.5709481239319|0.0273|9911/10000 (99%)|
|LeakyReLu|417.361878156662|0.0208|9933/10000 (99%)|

#### 卷积层数与全连接层数

- 原始三层

|**Total time (sec.)**|**Average loss**|**Accuracy**|
|:--:|:--:|:--:|
|431.2082631587982|0.0228|9928/10000 (99%)|

- 四层

|**Total time (sec.)**|**Average loss**|**Accuracy**|
|:--:|:--:|:--:|
|426.9151015281677|0.0191|9941/10000 (99%)|

- 五层

|**Total time (sec.)**|**Average loss**|**Accuracy**|
|:--:|:--:|:--:|
|884.6015954017639|0.0176|9940/10000 (99%)|

#### 添加残差模块

|**Total time (sec.)**|**Average loss**|**Accuracy**|
|:--:|:--:|:--:|
|1421.7657163143158|0.0175|9949/10000 (99%)|

#### 改变Loss计算方式

- 分别修改forward函数、train和test函数，将log\_softmax修改为softmax，即在计算loss时需要将F.nll\_loss修改为F.cross\_entropy；同时需要修改模型的前向传播函数中的最后一行代码，改为使用softmax函数。

|**Loss**|**Total time (sec.)**|**Average loss**|**Accuracy**|
|:--:|:--:|:--:|:--:|
|NLL loss|431.2082631587982|0.0228|9928/10000 (99%)|
|Softmax|709.2585787773132|1.4723|9889/10000 (99%)|

#### batch-size调整

- 不改变test的batch size

|**batch-size**|**Total time (sec.)**|**Average loss**|**Accuracy**|
|:--:|:--:|:--:|:--:|
|64|431.2082631587982|0.0228|9928/10000 (99%)|
|32|552.1606893539429|0.0210|9930/10000 (99%)|
|128|369.645516872406|0.0195|9934/10000 (99%)|
|256|334.3783414363861|0.0223|9920/10000 (99%)|

#### 总结

由于CNN对于mnist本身的accuary就比较高，达到99%，所以许多参数调整得出的结果都比较类似。比较明显的有Sigmoid作为激活函数时效果较差，以及网络层数增加与batch\_size的减小会显著增加运行时间。

### 代码

```python
# 以下代码实现了一个简单的3层 卷积神经网络
# 代码来源：https://github.com/pytorch/examples/tree/master/mnist

from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR
import time


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = stride
        self.conv1 = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False,
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(
            out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.stride = stride
        self.downsample = nn.Sequential(
            nn.Conv2d(
                in_channels, out_channels, kernel_size=1, stride=stride, bias=False
            ),
            nn.BatchNorm2d(out_channels),
        )

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)

        if self.stride != 1 or self.in_channels != self.out_channels:
            identity = self.downsample(x)
        out += identity
        out = self.relu(out)
        return out


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # nn.Conv2d() 参数依次代表： in_channnel, out_channel, kernel_size, stride
        # nn.Conv2d() 表示一个卷积层
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.conv3 = nn.Conv2d(64, 64, 3, 1)
        # self.res1 = ResidualBlock(32, 64)
        # self.res2 = ResidualBlock(64, 64)
        # dropout 比较有效的缓解过拟合的发生
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)

        # nn.Lineer() 全连接层 ，一般作为输出层，得到分类概率。fc2: 10 表示类别数
        self.fc1 = nn.Linear(7744, 512)
        self.fc2 = nn.Linear(512, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)  # relu= max(0, x), 是一个非线性激活函数
        x = self.conv2(x)
        x = F.relu(x)
        x = self.conv3(x)
        x = F.relu(x)
        # x = self.res1(x)
        # x = self.res2(x)
        x = F.max_pool2d(x, 2)  # max_pool2d 最大池化层
        x = self.dropout1(x)
        x = torch.flatten(x, 1)  # torch.flatten(x, start_dim, end_dim) 展平tensor x
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        output = F.softmax(x, dim=1)  # 这一步计算分类概率
        # output = F.log_softmax(x, dim=1)
        return output


def train(args, model, device, train_loader, optimizer, epoch):
    """
    该函数的作用: 训练网络模型
        args: 参数对象
        model: 网络模型
        device: CPU or GPU
        train_loader： 加载训练图片
        optimizer： 优化器
        epoch：训练次数
    """
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.cross_entropy(output, target)
        # loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print(
                "Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}".format(
                    epoch,
                    batch_idx * len(data),
                    len(train_loader.dataset),
                    100.0 * batch_idx / len(train_loader),
                    loss.item(),
                )
            )
            if args.dry_run:
                break


def test(model, device, test_loader):
    """
    该函数的作用: 测试网络模型
        args: 参数对象
        model: 网络模型
        train_loader：加载测试图片
    """
    model.eval()  # 让 model 进入测试模式，不能省略
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            # sum up batch loss
            # test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
            test_loss += F.cross_entropy(output, target, reduction="sum").item()
            # get the index of the max log-probability
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print(
        "\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n".format(
            test_loss,
            correct,
            len(test_loader.dataset),
            100.0 * correct / len(test_loader.dataset),
        )
    )


def main():
    # Training settings
    parser = argparse.ArgumentParser(description="PyTorch MNIST Example")
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        metavar="N",
        help="input batch size for training (default: 64)",
    )
    parser.add_argument(
        "--test-batch-size",
        type=int,
        default=1000,
        metavar="N",
        help="input batch size for testing (default: 1000)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=14,
        metavar="N",
        help="number of epochs to train (default: 14)",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1,
        metavar="LR",
        help="learning rate (default: 1.0)",
    )
    parser.add_argument(
        "--gamma",
        type=float,
        default=0.7,
        metavar="M",
        help="Learning rate step gamma (default: 0.7)",
    )
    parser.add_argument(
        "--no-cuda", action="store_true", default=False, help="disables CUDA training"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        default=False,
        help="quickly check a single pass",
    )
    parser.add_argument(
        "--seed", type=int, default=1, metavar="S", help="random seed (default: 1)"
    )
    parser.add_argument(
        "--log-interval",
        type=int,
        default=10,
        metavar="N",
        help="how many batches to wait before logging training status",
    )
    parser.add_argument(
        "--save-model",
        action="store_true",
        default=False,
        help="For Saving the current Model",
    )
    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()

    torch.manual_seed(args.seed)

    device = torch.device("cuda" if use_cuda else "cpu")

    train_kwargs = {"batch_size": args.batch_size}
    test_kwargs = {"batch_size": args.test_batch_size}
    if use_cuda:
        cuda_kwargs = {"num_workers": 1, "pin_memory": True, "shuffle": True}
        train_kwargs.update(cuda_kwargs)
        test_kwargs.update(cuda_kwargs)

    # 数据处理，转换tensor， 归一化
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
    )

    # '../data' 为数据的存放路径
    dataset1 = datasets.MNIST("../data", train=True, download=True, transform=transform)
    dataset2 = datasets.MNIST("../data", train=False, transform=transform)
    train_loader = torch.utils.data.DataLoader(dataset1, **train_kwargs)
    test_loader = torch.utils.data.DataLoader(dataset2, **test_kwargs)

    model = Net().to(device)
    optimizer = optim.Adadelta(model.parameters(), lr=args.lr)

    scheduler = StepLR(optimizer, step_size=1, gamma=args.gamma)
    for epoch in range(1, args.epochs + 1):
        train(args, model, device, train_loader, optimizer, epoch)
        test(model, device, test_loader)
        scheduler.step()

    if args.save_model:
        torch.save(model.state_dict(), "mnist_cnn.pt")


if __name__ == "__main__":
    start = time.time()
    main()  # 执行入口函数
    print("Total time: ", time.time() - start, "seconds")

```

##### 附：

运行设备GPU：RTX 3060
