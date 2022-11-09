import os
from os import listdir
from os.path import join
from PIL import Image
from torch.utils.data.dataset import Dataset
from torchvision.transforms import Compose, ToTensor, ToPILImage, CenterCrop, Resize
to_tensor = ToTensor()

def is_image_file(filename):
    return any(filename.endswith(extension) for extension in ['.png', '.jpg', '.jpeg', '.PNG', '.JPG', '.JPEG'])

def get_paths_from_images(path):
    assert os.path.isdir(path), '{:s} is not a valid directory'.format(path)
    images = []
    for dirpath, _, fnames in sorted(os.walk(path)):
        for fname in sorted(fnames):
            if is_image_file(fname):
                img_path = os.path.join(dirpath, fname)
                images.append(img_path)
    assert images, '{:s} has no valid image file'.format(path)
    return sorted(images)

class TrainDataset(Dataset):
    def __init__(self, dataroot, l_resolution=56, r_resolution=224):
        super(TrainDataset, self).__init__()
        self.hr_path = get_paths_from_images('{}/hr_{}'.format(dataroot, r_resolution))
        self.lr_path = get_paths_from_images('{}/sr_{}_{}'.format(dataroot, l_resolution, r_resolution))

    def __getitem__(self, index):
        hr_image = to_tensor(Image.open(self.hr_path[index]))
        lr_image = to_tensor(Image.open(self.lr_path[index]))
        return lr_image, hr_image

    def __len__(self):
        return len(self.hr_path)


class ValDataset(Dataset):
    def __init__(self, dataroot, l_resolution=56, r_resolution=224):
        super(ValDataset, self).__init__()
        self.hr_path = get_paths_from_images('{}/hr_{}'.format(dataroot, r_resolution))
        self.lr_path = get_paths_from_images('{}/sr_{}_{}'.format(dataroot, l_resolution, r_resolution))

    def __getitem__(self, index):
        hr_image = to_tensor(Image.open(self.hr_path[index]))
        lr_image = to_tensor(Image.open(self.lr_path[index]))
        return lr_image, hr_image

    def __len__(self):
        return len(self.hr_path)


class TestDatasetFromFolder(Dataset):
    def __init__(self, dataset_dir, upscale_factor):
        super(TestDatasetFromFolder, self).__init__()
        self.lr_path = dataset_dir + '/SRF_' + str(upscale_factor) + '/data/'
        self.hr_path = dataset_dir + '/SRF_' + str(upscale_factor) + '/target/'
        self.upscale_factor = upscale_factor
        self.lr_filenames = [join(self.lr_path, x) for x in listdir(self.lr_path) if is_image_file(x)]
        self.hr_filenames = [join(self.hr_path, x) for x in listdir(self.hr_path) if is_image_file(x)]

    def __getitem__(self, index):
        image_name = self.lr_filenames[index].split('/')[-1]
        lr_image = Image.open(self.lr_filenames[index])
        w, h = lr_image.size
        hr_image = Image.open(self.hr_filenames[index])
        hr_scale = Resize((self.upscale_factor * h, self.upscale_factor * w), interpolation=Image.BICUBIC)
        hr_restore_img = hr_scale(lr_image)
        return image_name, ToTensor()(lr_image), ToTensor()(hr_restore_img), ToTensor()(hr_image)

    def __len__(self):
        return len(self.lr_filenames)
