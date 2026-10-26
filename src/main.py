from abc import ABC, abstractmethod
import math
from matrix import Matrix
from vector import Vector

class Optimizer:
    class Model(ABC):
        @abstractmethod
        def step():
            pass

        @abstractmethod
        def zero_grad():
            pass

        