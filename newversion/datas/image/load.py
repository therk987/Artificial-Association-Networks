import torchvision.transforms as transforms
from torchvision.datasets import MNIST


def MNIST_DATA(root, download = True, resize_shape=(32,32)):
    transform = transforms.Compose([
                transforms.Resize(resize_shape),
                transforms.ToTensor(),
    ])

    mnist_train = MNIST(root = root,
                                 train = True,
                                 transform = transform,
                                 download = download)
    mnist_test = MNIST(root = root,
                                train = False,
                                transform = transform,
                                download = download)
    return mnist_train, mnist_test
