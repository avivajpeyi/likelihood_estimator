from abc import ABC, abstractmethod
from typing import Optional

import numpy as np


class DataContainer(ABC):
    """Abstract base class for data models

    Attributes:
        size: length of the data
        x: the numpy array of x values of the data
        y: the numpy array of the y values of the data
    """

    # Initializer / Instance Attributes
    def __init__(
        self, size: int, x: Optional[np.array] = None, y: Optional[np.array] = None
    ):
        self.size = size
        self.x = x if x is not None else np.zeros(self.size)
        self.y = y if y is not None else np.zeros(self.size)

    def __repr__(self):
        return self.__str__()

    # string
    def __str__(self):
        return str('{}:\nx: {}\ny: {}'.format(self.__class__.__name__, self.x, self.y))

    @staticmethod
    @abstractmethod
    def model(*args, **kwargs) -> np.array:
        pass

    @classmethod
    def from_dict(cls, data_dict: dict):
        return cls(
            size=data_dict.get('size', 0),
            x=data_dict.get('x', []),
            y=data_dict.get('y', []),
        )
