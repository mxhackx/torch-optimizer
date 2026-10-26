from abc import ABC, abstractmethod
from vector import Vector
from matrix import Matrix
import math

class ActivationFunctionError(Exception):
    pass

class Activation:

    class Model(ABC):
        @abstractmethod
        def _forward(self, x: int | float) -> int | float:
            pass

        @abstractmethod
        def _derivative(self, x: int | float) -> int | float:
            pass

        def forward(self, x: int | float) -> int | float:
            if isinstance(x, bool) or not isinstance(x, (int, float)):
                raise ActivationFunctionError(
                    "input must be of type int or float"
                )
            return self._forward(x)

        def derivative(self, x: int | float) -> int | float:
            if isinstance(x, bool) or not isinstance(x, (int, float)):
                raise ActivationFunctionError(
                    "input must be of type int or float"
                )
            return self._derivative(x)

        def forward_vector(self, x: Vector) -> Vector:
            if not isinstance(x, Vector):
                raise ActivationFunctionError(
                    "input must be of type Vector"
                )
            if x.empty():
                raise ActivationFunctionError("The vector is empty")

            return Vector([self._forward(x[i]) for i in range(len(x))])

        def derivative_vector(self, x: Vector) -> Vector:
            if not isinstance(x, Vector):
                raise ActivationFunctionError(
                    "input must be of type Vector"
                )
            if x.empty():
                raise ActivationFunctionError("The vector is empty")

            return Vector([self._derivative(x[i]) for i in range(len(x))])

        def forward_matrix(self, x: Matrix) -> Matrix:
            if not isinstance(x, Matrix):
                raise ActivationFunctionError(
                    "input must be of type Matrix"
                )
            if x.rows == 0 or x.cols == 0:
                raise ActivationFunctionError("The matrix is empty")

            return Matrix([
                [self._forward(x[i][j]) for j in range(x.cols)]
                for i in range(x.rows)
            ])

        def derivative_matrix(self, x: Matrix) -> Matrix:
            if not isinstance(x, Matrix):
                raise ActivationFunctionError(
                    "input must be of type Matrix"
                )
            if x.rows == 0 or x.cols == 0:
                raise ActivationFunctionError("The matrix is empty")

            return Matrix([
                [self._derivative(x[i][j]) for j in range(x.cols)]
                for i in range(x.rows)
            ])

        def __call__(self, x: int | float) -> int | float:
            return self.forward(x)

    class TensorModel(ABC):
        @abstractmethod
        def _forward(self, x: Vector | Matrix) -> Vector | Matrix:
            pass

        @abstractmethod
        def _derivative(self, x: Vector | Matrix) -> Vector | Matrix:
            pass

        def forward(self, x: Vector | Matrix) -> Vector | Matrix:
            if not isinstance(x, (Vector, Matrix)):
                raise ActivationFunctionError("input must be of type Vector or Matrix")
            return self._forward(x)

        def derivative(self, x: Vector | Matrix) -> Vector | Matrix:
            if not isinstance(x, (Vector, Matrix)):
                raise ActivationFunctionError("input must be of type Vector or Matrix")
            return self._derivative(x)

        def __call__(self, x: Vector | Matrix) -> Vector | Matrix:
            return self.forward(x)

    class Identity(Model):
        def _forward(self, x: int | float) -> int | float:
            return x

        def _derivative(self, x: int | float) -> int | float:
            return 1

    class ReLU(Model):
        def _forward(self, x: int | float) -> int | float:
            return max(0, x)

        def _derivative(self, x: int | float) -> int | float:
            return 1 if x > 0 else 0

    class Sigmoid(Model):
        def _forward(self, x: int | float) -> int | float:
            if x >= 0:
                return 1 / (1 + math.exp(-x))
            return math.exp(x) / (1 + math.exp(x))

        def _derivative(self, x: int | float) -> int | float:
            deriv = 0
            if x <= 0:
                try:
                    deriv = 1 / math.exp(x)
                except ZeroDivisionError:
                    return 0
            else:
                deriv = math.exp(-x)
            return deriv / (1 + deriv) ** 2

    class Tanh(Model):
        def _forward(self, x: int | float) -> int | float:
            if x >= 0:
                return (1 - math.exp(-2 * x)) / (1 + math.exp(-2 * x))
            return -(1 - math.exp(2 * x)) / (1 + math.exp(2 * x))

        def _derivative(self, x: int | float) -> int | float:
            return 1 - self._forward(x) ** 2

    class Leaky_ReLU(Model):
        def __init__(self, alpha = 1.0):
            self.alpha = alpha

        def _forward(self, x: int | float) -> int | float:
            return x if x >= 0 else self.alpha * x

        def _derivative(self, x: int | float) -> int | float:
            return 1 if x >= 0 else self.alpha

    class Softmax(TensorModel):
        def _forward(self, x: Vector) -> Vector:
            if not isinstance(x, Vector):
                raise ActivationFunctionError("x must be of type Vector")
            if x.empty():
                raise ActivationFunctionError("The vector is empty")
            max_x = max(x)
            vector_x = Vector([math.exp(ix - max_x) for ix in x])
            sum_x = sum(vector_x)
            return Vector([ix / sum_x for ix in vector_x])

        def _derivative(self, x: Vector) -> Matrix:
            if not isinstance(x, Vector):
                raise ActivationFunctionError("x must be of type Vector")
            if x.empty():
                raise ActivationFunctionError("The vector is empty")
            v_x = self._forward(x)
            len_x = len(v_x)
            matrix = []
            for i in range(len_x):
                matrix.append([])
                for j in range(len_x):
                    if i == j:
                        matrix[i].append(v_x[i] * (1 - v_x[i]))
                    else:
                        matrix[i].append(-v_x[i] * v_x[j])
            return Matrix(matrix)

    class LogSoftmax(TensorModel):
        def _forward(self, x: Vector) -> Vector:
            if not isinstance(x, Vector):
                raise ActivationFunctionError("x must be of type Vector")
            if x.empty():
                raise ActivationFunctionError("The vector is empty")
            max_x = max(x)
            vector_x = Vector([math.exp(ix - max_x) for ix in x])
            sum_x = math.log(sum(vector_x))
            return Vector([ix - max_x - sum_x for ix in x])

        def _derivative(self, x: Vector) -> Matrix:
            if not isinstance(x, Vector):
                raise ActivationFunctionError("x must be of type Vector")
            if x.empty():
                raise ActivationFunctionError("The vector is empty")

            s = Activation.Softmax().forward(x)
            n = len(s)
            matrix = []

            for i in range(n):
                row = []
                for j in range(n):
                    if i == j:
                        row.append(1 - s[j])
                    else:
                        row.append(-s[j])
                matrix.append(row)

            return Matrix(matrix)

    class ELU(Model):
        def __init__(self, alpha = 1.0):
            self.alpha = alpha

        def _forward(self, x):
            return x if x >= 0 else self.alpha * (math.exp(x) - 1)

        def _derivative(self, x):
            return 1 if x >= 0 else self.alpha * math.exp(x)

    class GeLU(Model):
        def _forward(self, x: int | float) -> int | float:
            return x / 2 * (1 + math.erf(x / math.sqrt(2)))

        def _derivative(self, x: int | float) -> int | float:
            return 1 / 2 * (1 + math.erf(x / math.sqrt(2))) + x / math.sqrt(2 * math.pi) * math.exp(-(x ** 2)/2)
