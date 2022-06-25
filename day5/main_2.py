import numpy as np
from PIL import Image
import paddle
import paddle.vision.transforms as T

paddle.set_device('cpu')


def crop(image, region):
    cropped_image = T.crop(image, *region)
    return cropped_image


class CenterCrop():
    def __init__(self, size):
        self.size = size
    
    def __call__(self, image):
        w, h = image.size
        ch, cw = self.size
        crop_top = int(round(h-ch) / 2.)
        crop_left = int(round(w-cw) / 2.)
        return crop(image, (crop_top, crop_left, ch, cw))


class Resize():
    def __init__(self, size):
        self.size = size
    
    def __call__(self, image):
        return T.resize(image, self.size)

class ToTensor():
    def __init__(self) -> None:
        pass

    def __call__(self, image):
        image = paddle.to_tensor(np.array(image))
        if image.dtype == paddle.uint8:
            image = paddle.cast(image, dtype='float32') / 255.
        image = image.transpose([2, 0, 1])
        return image


class Compose():
    def __init__(self, transforms):
        self.transforms = transforms
    
    def __call__(self, image):
        for t in self.transforms:
            image = t(image)
        return image


def main():
    img = Image.open('../data/image.jpg')
    print(np.array(img).shape)
    transform = Compose([Resize([256, 256]),
                        CenterCrop([176, 176]),
                        ToTensor()])

    out = transform(img)
    print(out)
    print(out.shape)

    # save image
    image = out.transpose([1, 2, 0]).cpu().numpy()*255.
    print(image.shape)
    image = Image.fromarray(image.astype('uint8'))
    image.save('../data/cropped_image.jpg')
    image.show()

if __name__ == '__main__':
    main()