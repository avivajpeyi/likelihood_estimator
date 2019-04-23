from typing import List, Optional


class LikelihoodEstimator(object):
    """Blah blah

    Attributes:
        dict: adasadsd
    """

    # Initializer / Instance Attributes
    def __init__(self, class_attribute: Optional[List[int]] = [1]):
        self.class_attribute = class_attribute

    def __repr(self):
        return self.__str__()

    # string
    def __str__(self):
        return str(self.class_attribute)

    # protected method
    def _process(self, idx):
        self.class_attribute[idx] = self.class_attribute[idx] * 2

    # instance method
    def lst(self, min, max):
        return [self._process(idx=val) for val in self.class_attribute[min:max]]

    @classmethod
    def factory_method_for_class(cls, class_attribute):
        return cls(class_attribute)

    @staticmethod
    def calculate(a, b):
        return a + b
