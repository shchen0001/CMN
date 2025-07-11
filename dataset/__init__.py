from .cars import Cars
from .cub import CUBirds
from .SOP import SOP
from .aircraft import Aircraft
from .nabirds import NABirds
from .import utils
from .base import BaseDataset


_type = {
    'cars': Cars,
    'cub': CUBirds,
    'SOP': SOP,
    'nabirds': NABirds,
    'air': Aircraft
}

def load(name, root, mode, transform = None):
    return _type[name](root = root, mode = mode, transform = transform)
    
