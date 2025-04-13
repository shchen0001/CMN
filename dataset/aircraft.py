from .base import *
import scipy.io

class Aircraft(BaseDataset):
    def __init__(self, root, mode, transform = None):
        self.root = root + '/Aircraft_dataset/fgvc-aircraft-2013b/'
        self.mode = mode
        self.transform = transform
        if self.mode == 'train':
            self.classes = range(0,50)
        elif self.mode == 'eval':
            self.classes = range(50,100)
        
        BaseDataset.__init__(self, self.root, self.mode, self.transform)
    
        variants = []
        with open(os.path.join(self.root, 'data/variants.txt'), 'r') as f:
            for line in f:
                variants.append(line.strip())
        file_list = []
        label = []
        with open(os.path.join(self.root, 'data/images_variant_trainval.txt'), 'r') as f:
            for line in f:
                line = line.strip()
                seven_digits = line[:7]
                variant = line[8:]
                file_list.append(os.path.join(self.root, 'data/images', seven_digits+'.jpg'))
                label.append(variants.index(variant))
        with open(os.path.join(self.root, 'data/images_variant_test.txt'), 'r') as f:
            for line in f:
                line = line.strip()
                seven_digits = line[:7]
                variant = line[8:]
                file_list.append(os.path.join(self.root, 'data/images', seven_digits+'.jpg'))
                label.append(variants.index(variant))    
        index = 0
        for im_path, y in zip(file_list, label):
            if y in self.classes: # choose only specified classes
                self.im_paths.append(os.path.join(self.root, im_path))
                self.ys.append(y)
                self.I += [index]
                index += 1
