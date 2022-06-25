# -*-coding:utf-8-*-
# author lyl
from PIL import Image
import numpy as np
import paddle


def main():
    # 1. Create a Tensor
    t = paddle.zeros([3, 3])
    print(t)
    # 2. Create a Random Tensor
    t = paddle.randn([5, 3])
    print(t)
    # 3. Create a tensor from Image ./724.jpg 28x28
    img = np.array(Image.open(''))
    for i in range(28):
        for j in range(28):
            print(f"{img[i, j]:03}", end=' ')
        print()
    # 4.print tensor type and dtype of tensor
    t = paddle.to_tensor(img)
    print(type(t))
    print(t.dtype)

    # 5. transpose image tensor
    print(t.transpose([1, 0]))

if __name__ == '__main__':
    main()