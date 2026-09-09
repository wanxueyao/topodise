"""Image transforms. Output single-channel tensors normalized to [-1, 1]."""
import torchvision.transforms as T
import torchvision.transforms.functional as TF


def _to_gray_minus1_1(img):
    img = TF.to_grayscale(img, num_output_channels=1)
    img = TF.to_tensor(img)            # [0,1]
    return img * 2.0 - 1.0


class _GrayNorm:
    def __call__(self, img):
        return _to_gray_minus1_1(img)


def build_pretrain_transform(image_size: int = 256, strength: float = 0.5) -> T.Compose:
    return T.Compose([
        T.Resize(int(image_size * 1.1)),
        T.RandomCrop(image_size),
        T.RandomHorizontalFlip(0.5),
        T.RandomAffine(degrees=15 * strength, translate=(0.05 * strength, 0.05 * strength),
                       scale=(1 - 0.1 * strength, 1 + 0.1 * strength)),
        _GrayNorm(),
    ])


def build_eval_transform(image_size: int = 256) -> T.Compose:
    return T.Compose([
        T.Resize(image_size),
        T.CenterCrop(image_size),
        _GrayNorm(),
    ])
