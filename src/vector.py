import math


class VectorError(Exception):
    pass


class Vector:
    def __init__(self, vector: list):
        if not isinstance(vector, list):
            raise VectorError("The first parameter must be a list")
        for i in vector:
            if not isinstance(i, (float, int)):
                raise VectorError("There is some unknown type in your list")
        self.vector = vector.copy()
        self.len = len(vector)
    @staticmethod
    def epsilon(v: float):
        if not isinstance(v, (float, int)):
            raise VectorError("Unknown type")
        return abs(v) < 1e-12

    def __add__(self, vector):
        if not isinstance(vector, Vector):
            raise VectorError("Not a Vector")
        if len(vector.vector) != len(self.vector):
            raise VectorError("Not the Same length")
        return Vector([i + j for i, j in zip(self.vector, vector.vector)])

    def __sub__(self, vector: list):
        if not isinstance(vector, Vector):
            raise VectorError("Not a Vector")
        if len(vector) != len(self):
            raise VectorError("Not the Same length")
        return Vector([i - j for i, j in zip(self.vector, vector.vector)])

    def __mul__(self, v):
        if isinstance(v, (float, int)):
            return Vector([i * v for i in self.vector])
        raise VectorError("Unknow type")

    def __truediv__(self, v):
        if not isinstance(v, (int, float)):
            raise VectorError("Incorrect type")
        if self.epsilon(v):
            raise VectorError("Zero Division error")
        return Vector([i / v for i in self.vector])

    def __pow__(self, v):
        if not isinstance(v, (int, float)):
            raise VectorError("Incorrect type")
        return Vector([i**v for i in self.vector])

    def __floordiv__(self, v):
        if not isinstance(v, (int, float)):
            raise VectorError("Incorrect type")
        if self.epsilon(v):
            raise VectorError("Zero Division error")
        return Vector([i // v for i in self.vector])

    def __mod__(self, v):
        if not isinstance(v, (int, float)):
            raise VectorError("Incorrect type")
        if self.epsilon(v):
            raise VectorError("Zero division error")
        return Vector([i % v for i in self.vector])

    def __neg__(self):
        return Vector([-i for i in self.vector])

    def __radd__(self, vector):
        if not isinstance(vector, Vector):
            raise VectorError("Not a Vector")
        if len(vector) != len(self.vector):
            raise VectorError("Not the Same length")
        return Vector([i + j for i, j in zip(self.vector, vector.vector)])

    def __rsub__(self, vector):
        if not isinstance(vector, Vector):
            raise VectorError("Not a Vector")
        if len(vector) != len(self.vector):
            raise VectorError("Not the Same length")
        return Vector([j - i for i, j in zip(self.vector, vector.vector)])

    def __rmul__(self, v):
        if isinstance(v, (float, int)):
            return Vector([i * v for i in self.vector])
        raise VectorError("Unknow type")

    def __rtruediv__(self, v):
        if not isinstance(v, (int, float)):
            raise VectorError("Incorrect type")
        if any(self.epsilon(i) for i in self.vector):
            raise VectorError("Zero division error")
        return Vector([v / i for i in self.vector])

    def __rpow__(self, v):
        if not isinstance(v, (int, float)):
            raise VectorError("Incorrect type")
        return Vector([v**i for i in self.vector])

    def __rfloordiv__(self, v):
        if not isinstance(v, (int, float)):
            raise VectorError("Incorrect type")
        if any(self.epsilon(i) for i in self.vector):
            raise VectorError("Zero division error")
        return Vector([v // i for i in self.vector])

    def __rmod__(self, v):
        if not isinstance(v, (int, float)):
            raise VectorError("Incorrect type")
        if any(self.epsilon(i) for i in self.vector):
            raise VectorError("Zero division error")
        return Vector([v % i for i in self.vector])

    def __iadd__(self, vector):
        if not isinstance(vector, Vector):
            raise VectorError("Not a Vector")
        if len(vector) != len(self.vector):
            raise VectorError("Not the Same length")
        self.vector = [i + j for i, j in zip(self.vector, vector.vector)]
        return self

    def __isub__(self, vector: list):
        if not isinstance(vector, Vector):
            raise VectorError("Not a Vector")
        if len(vector) != len(self.vector):
            raise VectorError("Not the Same length")
        self.vector = [i - j for i, j in zip(self.vector, vector.vector)]
        return self

    def __imul__(self, v):
        if isinstance(v, (float, int)):
            self.vector = [i * v for i in self.vector]
            return self
        raise VectorError("Unknow type")

    def __itruediv__(self, v):
        if not isinstance(v, (int, float)):
            raise VectorError("Incorrect type")
        if self.epsilon(v):
            raise VectorError("Zero Division error")
        self.vector = [i / v for i in self.vector]
        return self

    def __ipow__(self, v):
        if not isinstance(v, (int, float)):
            raise VectorError("Incorrect type")
        self.vector = [i**v for i in self.vector]
        return self

    def __ifloordiv__(self, v):
        if not isinstance(v, (int, float)):
            raise VectorError("Incorrect type")
        if self.epsilon(v):
            raise VectorError("Zero Division error")
        self.vector = [i // v for i in self.vector]
        return self

    def __imod__(self, v):
        if not isinstance(v, (int, float)):
            raise VectorError("Incorrect type")
        if self.epsilon(v):
            raise VectorError("Zero division error")
        self.vector = [i % v for i in self.vector]
        return self

    def __iter__(self):
        return iter(self.vector)

    def __getitem__(self, index):
        return self.vector[index]

    def __str__(self):
        s = "Vector: \n[" + " ".join(map(str, self.vector)) + " ]"
        return s

    def __len__(self):
        return self.len

    def dot(self, v):
        if not isinstance(v, Vector):
            raise VectorError("That's not a vector")
        if len(v) != len(self.vector):
            raise VectorError("Not the Same length")
        return sum([i * j for i, j in zip(self.vector, v.vector)])

    def norm(self):
        return math.sqrt(self.dot(self))

    def dist(self, v):
        if not isinstance(v, Vector):
            raise VectorError("Incorrect type")
        return math.sqrt(sum([(i - j) ** 2 for i, j in zip(self.vector, v.vector)]))

    def normalize(self):
        n = self.norm()
        if self.epsilon(n):
            raise VectorError("Division by zero in normalize")
        return Vector([i / n for i in self.vector])

    def __eq__(self, v):
        if not isinstance(v, Vector):
            return False
        return len(self) == len(v) and all(
            math.isclose(i, j) for i, j in zip(self.vector, v.vector)
        )

    def __ne__(self, v):
        return not self.__eq__(v)

    def __setitem__(self, i, v):
        if not isinstance(v, (int, float)):
            raise VectorError("Incorrect type")
        self.vector[i] = v

    def __contains__(self, v):
        if not isinstance(v, (int, float)):
            return False
        return v in self.vector

    def cos(self, v):
        if not isinstance(v, Vector):
            raise VectorError("Not a vector")
        a = self.dot(v)
        b = self.norm() * v.norm()
        if self.epsilon(b):
            raise VectorError("Division by zero Error")
        return a / b

    def angle(self, v):
        return math.acos(self.cos(v))

    def project(self, v):
        a = v.dot(v)
        b = self.dot(v)
        if self.epsilon(a):
            raise VectorError("Division by zero")
        return v * (b / a)

    def cross(self, v):
        if not isinstance(v, Vector) or (len(self.vector) != len(v) or len(v) != 3):
            raise VectorError("Error while computing vectoriel")
        a = self.vector[1] * v.vector[2] - self.vector[2] * v.vector[1]
        b = self.vector[2] * v.vector[0] - self.vector[0] * v.vector[2]
        c = self.vector[0] * v.vector[1] - self.vector[1] * v.vector[0]
        return Vector([a, b, c])

    def hadamard(self, v):
        if not isinstance(v, Vector) or (len(self.vector) != len(v)):
            raise VectorError("Error while computing vectoriel")
        return Vector([i * j for i, j in zip(self.vector, v.vector)])

    def div(self, v):
        if not isinstance(v, Vector) or (len(self.vector) != len(v)):
            raise VectorError("Error while computing vectoriel")
        for i in v.vector:
            if self.epsilon(i):
                raise VectorError("Division by zero")
        return Vector([i / j for i, j in zip(self.vector, v.vector)])

    def clamp(self, mi, ma):
        return Vector([max(mi, min(i, ma)) for i in self.vector])

    def reflex(self, n):
        if not isinstance(n, Vector) or len(n) != len(self.vector):
            raise VectorError("Vector error")
        a = n.normalize()
        a = self.dot(a) * a
        a = 2 * a
        return self - a

    def __copy__(self):
        return self.__class__(self.vector)

    @classmethod
    def null_vector(cls, v):
        if not isinstance(v, int):
            raise VectorError("That's not an integer")
        return cls([0] * v)

    def empty(self):
        return len(self.vector) == 0

    def __sum__(self):
        return sum(self.vector)

if __name__ == "__main__":
    v = Vector([1, 0.00001, 0.00001])
    print(v.normalize())
