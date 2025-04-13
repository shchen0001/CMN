from .base import *

class NABirds(BaseDataset):
    def __init__(self, root, mode = 'eval', transform = None):
        self.root = root + '/nabirds/'
        self.mode = mode
        self.transform = transform
        self.classes = range(0,1011)
        
        BaseDataset.__init__(self, self.root, self.mode, self.transform)
        img_txt = os.path.join(self.root, 'images.txt')
        image_class_labels_txt = os.path.join(self.root, 'image_class_labels.txt')
        class_name = open(os.path.join(self.root, 'classes.txt'))

        with open(img_txt) as f:
            index = 0
            for line in f:
                pieces = line.strip().split()
                self.I += [index]
                self.im_paths.append(os.path.join(self.root, 'images', pieces[1]))
                index += 1

        with open(image_class_labels_txt) as f:
            for line in f:
                pieces = line.strip().split()
                self.ys += [int(pieces[1])]