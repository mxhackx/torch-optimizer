import inspect
import math
import random

class MatrixError(Exception):
    pass


class Matrix:
    @staticmethod
    def is_matrix(matrix):
        if (
            not isinstance(matrix, list)
            or len(matrix) == 0
            or not isinstance(matrix[0], list)
        ):
            raise MatrixError("That's not a matrix")
        len_column = len(matrix[0])
        for i in matrix:
            if not isinstance(i, list) or len_column != len(i) or len(i) == 0:
                raise MatrixError("Not the same length in the matrix")
            for j in i:
                if not isinstance(j, (int, float)):
                    raise MatrixError("Unknown type in the matrix")
        return True

    @staticmethod
    def check_matrix(matrix):
        if not isinstance(matrix, Matrix):
            raise MatrixError("That's not a matrix")

    @staticmethod
    def _op_matrix(m1, m2, op):
        if m1.cols != m2.cols or m1.rows != m2.rows:
            raise MatrixError("Incorrect length")
        return Matrix([[op(k, l) for k, l in zip(i, j)] for i, j in zip(m1, m2)])

    @staticmethod
    def _op_matrix_zero(m1, m2, op):
        if m1.cols != m2.cols or m1.rows != m2.rows:
            raise MatrixError("Incorrect length")
        if any(any(Matrix.is_close(j, 0) for j in i) for i in m2):
            raise MatrixError("Divison by Zero Error")
        return Matrix([[op(k, l) for k, l in zip(i, j)] for i, j in zip(m1, m2)])

    @staticmethod
    def _op(m, x, op):
        if not isinstance(x, (int, float)):
            raise MatrixError("Unknown x type(should be int or float)")
        return Matrix([[op(j, x) for j in i] for i in m])

    @staticmethod
    def _op_zero(m, x, op):
        if not isinstance(x, (int, float)):
            raise MatrixError("Unknown x type(should be int or float)")
        if Matrix.is_close(x, 0):
            raise MatrixError("Division by zero error")
        return Matrix([[op(j, x) for j in i] for i in m])

    @staticmethod
    def _rop(m, x, op):
        if not isinstance(x, (int, float)):
            raise MatrixError("Unknown x type(should be int or float)")
        return Matrix([[op(x, j) for j in i] for i in m])

    @staticmethod
    def _rop_zero(m, x, op):
        if not isinstance(x, (int, float)):
            raise MatrixError("Unknown x type(should be int or float)")
        if any(any(Matrix.is_close(j, 0) for j in i) for i in m):
            raise MatrixError("Division by zero error")
        return Matrix([[op(x, j) for j in i] for i in m])

    @staticmethod
    def is_close(a, b):
        if not isinstance(a, (int, float)) or not isinstance(b, (int, float)):
            raise MatrixError("Incorrect type")
        return abs(a - b) < 1e-12

    def same_cols(self, matrix):
        Matrix.check_matrix(matrix)
        return matrix.cols == self.cols

    def same_rows(self, matrix):
        Matrix.check_matrix(matrix)
        return matrix.rows == self.rows

    def multipliable(self, matrix):
        Matrix.check_matrix(matrix)
        return self.cols == matrix.rows

    def rmultipliable(self, matrix):
        Matrix.check_matrix(matrix)
        return self.rows == matrix.cols

    def is_same_shape(self, matrix):
        Matrix.check_matrix(matrix)
        return (matrix.cols == self.cols) and (matrix.rows == self.rows)

    @classmethod
    def is_square(cls, matrix):
        cls.check_matrix(matrix)
        return matrix.cols == matrix.rows

    def __init__(self, matrix):
        Matrix.is_matrix(matrix)
        self.rows = len(matrix)
        self.cols = len(matrix[0])
        self.matrix = [i[:] for i in matrix]

    def __add__(self, matrix):
        if isinstance(matrix, (int, float)):
            return Matrix._op(self.matrix, matrix, lambda x, y: x + y)
        if not self.is_same_shape(matrix):
            raise MatrixError("Not the same size")
        return Matrix._op_matrix(self, matrix, lambda x, y: x + y)

    def __sub__(self, matrix):
        if isinstance(matrix, (int, float)):
            return Matrix._op(self.matrix, matrix, lambda x, y: x - y)
        if not self.is_same_shape(matrix):
            raise MatrixError("Not the same size")
        return Matrix._op_matrix(self, matrix, lambda x, y: x - y)

    def __mul__(self, x):
        if isinstance(x, Matrix):
            return Matrix._op_matrix(self, x, lambda a, b: a * b)
        if not isinstance(x, (int, float)):
            raise MatrixError("Not a float")
        return Matrix._op(self.matrix, x, lambda x, y: x * y)

    def __truediv__(self, x):
        if isinstance(x, Matrix):
            return Matrix._op_matrix_zero(self, x, lambda x, y: x / y)
        if not isinstance(x, (int, float)):
            raise MatrixError("Not a float")
        return Matrix._op_zero(self.matrix, x, lambda x, y: x / y)

    def __floordiv__(self, x):
        if isinstance(x, Matrix):
            return Matrix._op_matrix_zero(self, x, lambda x, y: x // y)
        if not isinstance(x, (int, float)):
            raise MatrixError("Not a float")
        return Matrix._op_zero(self.matrix, x, lambda x, y: x // y)

    def __mod__(self, x):
        if isinstance(x, Matrix):
            return Matrix._op_matrix_zero(self, x, lambda x, y: x % y)
        if not isinstance(x, (int, float)):
            raise MatrixError("Not a float")
        return Matrix._op_zero(self.matrix, x, lambda x, y: x % y)

    def __pow__(self, x):
        if isinstance(x, Matrix):
            return Matrix._op_matrix(self, x, lambda x, y: x**y)
        if not isinstance(x, (int, float)):
            raise MatrixError("Not a float")
        return Matrix._op(self.matrix, x, lambda x, y: x**y)

    def __radd__(self, matrix):
        if isinstance(matrix, (int, float)):
            return Matrix._op(self.matrix, matrix, lambda x, y: x + y)
        if not self.is_same_shape(matrix):
            raise MatrixError("Not the same size")
        return Matrix._op_matrix(matrix, self, lambda x, y: x + y)

    def __rsub__(self, matrix):
        if isinstance(matrix, (int, float)):
            return Matrix._rop(self.matrix, matrix, lambda x, y: x - y)
        if not self.is_same_shape(matrix):
            raise MatrixError("Not the same size")
        return Matrix._op_matrix(matrix, self, lambda x, y: x - y)

    def __rmul__(self, x):
        if isinstance(x, Matrix):
            return Matrix._op_matrix(x, self, lambda x, y: x * y)
        if not isinstance(x, (int, float)):
            raise MatrixError("Not a float")
        return Matrix._rop(self.matrix, x, lambda x, y: x * y)

    def __rtruediv__(self, x):
        if isinstance(x, Matrix):
            return Matrix._op_matrix_zero(x, self, lambda x, y: x / y)
        if not isinstance(x, (int, float)):
            raise MatrixError("Not a float")
        return Matrix._rop_zero(self.matrix, x, lambda x, y: x / y)

    def __rfloordiv__(self, x):
        if isinstance(x, Matrix):
            return Matrix._op_matrix_zero(x, self, lambda x, y: x // y)
        if not isinstance(x, (int, float)):
            raise MatrixError("Not a float")
        return Matrix._rop_zero(self.matrix, x, lambda x, y: x // y)

    def __rmod__(self, x):
        if isinstance(x, Matrix):
            return Matrix._op_matrix_zero(x, self, lambda x, y: x % y)
        if not isinstance(x, (int, float)):
            raise MatrixError("Not a float")
        return Matrix._rop_zero(self.matrix, x, lambda x, y: x % y)

    def __rpow__(self, x):
        if isinstance(x, Matrix):
            return Matrix._op_matrix(x, self, lambda x, y: x**y)
        if not isinstance(x, (int, float)):
            raise MatrixError("Not a float")
        return Matrix._rop(self.matrix, x, lambda x, y: x**y)

    def __iadd__(self, x):
        self.matrix = self.__add__(x).matrix
        return self

    def __isub__(self, x):
        self.matrix = self.__sub__(x).matrix
        return self

    def __imul__(self, x):
        self.matrix = self.__mul__(x).matrix
        return self

    def __itruediv__(self, x):
        self.matrix = self.__truediv__(x).matrix
        return self

    def __ifloordiv__(self, x):
        self.matrix = self.__floordiv__(x).matrix
        return self

    def __imod__(self, x):
        self.matrix = self.__mod__(x).matrix
        return self

    def __ipow__(self, x):
        self.matrix = self.__pow__(x).matrix
        return self

    def __iter__(self):
        return iter(self.matrix)
    def flatten(self):
        return [x for y in self.matrix for x in y]
    def __getitem__(self, i):
        if isinstance(i, tuple):
            if (i[0] < 0 or i[0] >= self.rows) or (i[1] < 0 or i[1] >= self.cols):
                raise MatrixError("Index out of range")
            if len(i) == 2:
                return self.matrix[i[0]][i[1]]
            raise MatrixError("Incorrect tuple")
        if i < 0 or i >= self.rows:
            raise MatrixError("Index out of range")
        return self.matrix[i]

    def __setitem__(self, i, k):
        if isinstance(i, int):
            if i < 0 or i >= self.rows:
                raise MatrixError("Index out of range")
            if (
                isinstance(k, list)
                and all(isinstance(j, (int, float)) for j in k)
                and len(k) == self.cols
            ):
                self.matrix[i] = k
            else:
                raise MatrixError("List incorrect length")
        else:
            if not isinstance(i, tuple):
                raise MatrixError("Not a tuple")
            if not isinstance(k, (int, float)):
                raise MatrixError("That should be a float or an integer")
            if len(i) != 2 or not all(isinstance(j, int) for j in i):
                raise MatrixError("Incorrect tuple type")
            if (i[0] < 0 or i[0] >= self.rows) or (i[1] < 0 or i[1] >= self.cols):
                raise MatrixError("Index out of range")
            self.matrix[i[0]][i[1]] = k

    def __eq__(self, matrix):
        if self.is_same_shape(matrix):
            return all(
                all(Matrix.is_close(k, l) for k, l in zip(i, j))
                for i, j in zip(self.matrix, matrix.matrix)
            )
        return False

    def __ne__(self, matrix):
        return not self.__eq__(matrix)

    def __contains__(self, value):
        if isinstance(value, list):
            return value in self.matrix
        if isinstance(value, Matrix):
            return value.matrix in self.matrix
        if isinstance(value, (int, float)):
            return any(any(Matrix.is_close(value, j) for j in i) for i in self.matrix)
        return False

    def shape(self):
        return (self.rows, self.cols)

    def copy(self):
        return Matrix(self.matrix)

    def transpose(self):
        return Matrix([list(i) for i in zip(*self.matrix)])

    def __matmul__(self, ma):
        if not self.multipliable(ma):
            raise MatrixError("Matrixes are not multipliable")
        matrix = ma.transpose()
        m = []
        for i in range(self.rows):
            m.append([])
            for j in range(matrix.rows):
                m[i].append(sum(matrix.matrix[j][k] * self.matrix[i][k]for k in range(self.cols)))
        return Matrix(m)

    @classmethod
    def eye(cls, x):
        if isinstance(x, int):
            return cls([[1 if i == j else 0 for j in range(x)] for i in range(x)])
        elif (
            isinstance(x, tuple)
            and len(x) == 2
            and isinstance(x[0], int)
            and isinstance(x[1], int)
            and x[0] >= 0
            and x[1] >= 0
        ):
            return cls([[1 if i == j else 0 for j in range(x[1])] for i in range(x[0])])
        else:
            raise MatrixError("Incorrect matrix type")

    @classmethod
    def null(cls, x):
        if isinstance(x, int):
            return cls([[0 for j in range(x)] for i in range(x)])
        elif (
            isinstance(x, tuple)
            and len(x) == 2
            and isinstance(x[0], int)
            and isinstance(x[1], int)
            and x[0] >= 0
            and x[1] >= 0
        ):
            return cls([[0 for j in range(x[1])] for i in range(x[0])])
        else:
            raise MatrixError("Incorrect matrix type")

    def trace(self):
        if self.cols != self.rows:
            raise MatrixError("Cols and Rows should be equal")
        return sum(self.matrix[i][i] for i in range(self.rows))

    def dot(self, X):
        if (
            not isinstance(X, list)
            or len(X) != self.cols
            or any(not isinstance(i, (int, float)) for i in X)
        ):
            raise MatrixError("That's not a list")
        Y = [0] * self.rows
        for i in range(self.rows):
            Y[i] = sum(j * k for j, k in zip(self.matrix[i], X))
        return Matrix([Y])

    def __str__(self):
        return "Matrix: \n" + ",\n".join(
            ["[\t" + "\t".join(map(str, i)) + "\t]" for i in self.matrix]
        )

    def frobenius(self):
        return math.sqrt(sum(j * j for i in self.matrix for j in i))

    def norm1(self):
        return max(
            sum(abs(self.matrix[i][j]) for i in range(self.rows))
            for j in range(self.cols)
        )

    def norm_inf(self):
        return max(sum(abs(x) for x in row) for row in self.matrix)

    def normalize_frobenius(self):
        frobenius = self.frobenius()
        if Matrix.is_close(frobenius, 0):
            raise MatrixError("Frobenius norm is close to 0, can't normalize")
        return self / frobenius

    @classmethod
    def minor(cls, m, row, col):
        if not Matrix.is_square(m):
            raise MatrixError("Matrix should be a square")
        if not isinstance(row, int) or not isinstance(col, int):
            raise MatrixError("Row and Col should be an integer")
        matrix = []
        for i in range(m.rows):
            if i == row:
                continue
            new_row = []
            matrix.append(new_row)
            for j in range(m.cols):
                if j == col:
                    continue
                new_row.append(m.matrix[i][j])
        return Matrix(matrix)

    @classmethod
    def det(cls, matrix):
        if not cls.is_square(matrix):
            raise MatrixError("Matrix should be a square")
        if len(matrix.matrix) == 1:
            return matrix.matrix[0][0]
        elif len(matrix.matrix) == 2:
            return matrix.matrix[0][0] * matrix.matrix[1][1] - matrix.matrix[0][1] * matrix.matrix[1][0]
        return sum(matrix.matrix[0][i] * cls.cofactor(matrix, 0, i) for i in range(matrix.cols))

    @classmethod
    def cofactor(cls, matrix, i, j):
        Matrix.check_matrix(matrix)
        return (-1)**(i + j) * cls.det(cls.minor(matrix, i, j))

    def inverse(self):
        if not Matrix.is_square(self):
            raise MatrixError("Matrix should be a square")
        det = Matrix.det(self)
        if Matrix.is_close(det, 0):
            raise MatrixError("matrix is not inversible")
        matrix = []
        for i in range(self.rows):
            matrix.append([])
            for j in range(self.cols):
                matrix[i].append(Matrix.cofactor(self, i, j))
        matrix = Matrix(matrix).transpose()
        matrix.__itruediv__(det)
        return matrix

    @classmethod
    def solve(cls, a, b):
        if not cls.is_square(a):
            raise MatrixError("Matrix should be a square")
        if not Matrix.same_rows(a, b):
            raise MatrixError("The matrixes should have the same number of rows")
        if (b.cols != 1):
            raise MatrixError("B should be a matrix with a unique column")
        return a.inverse() @ b

    @classmethod
    def generate_invertible_matrix(cls, n):
        matrix = [[0 for _ in range(n)] for _ in range(n)]

        for i in range(n):
            for j in range(i, n):
                if i == j:
                    matrix[i][j] = random.randint(1, 10)
                else:
                    matrix[i][j] = random.randint(-5, 5)
        return Matrix(matrix)

    def gauss_jordan(self):
        if not Matrix.is_square(self):
            raise MatrixError("Matrix should be a square")
        eye = Matrix.eye(self.cols)
        matrix = self.copy()
        for j in range(self.cols):
            #Calcule du pivot
            pivot = matrix.matrix[j][j]
            pivot_row = j
            for k in range(j, self.rows):
                if (abs(matrix.matrix[k][j]) > abs(pivot)):
                    pivot = matrix.matrix[k][j]
                    pivot_row = k
            if pivot_row != j:
                matrix.matrix[j], matrix.matrix[pivot_row] = matrix.matrix[pivot_row], matrix.matrix[j]
                eye.matrix[j], eye.matrix[pivot_row] = eye.matrix[pivot_row], eye.matrix[j]
            pivot = matrix.matrix[j][j]
            if Matrix.is_close(pivot, 0):
                raise MatrixError("Pivot is null")
            #Matrix multiplication
            for i in range(self.cols):
                matrix.matrix[j][i] /= pivot
                eye.matrix[j][i] /= pivot
            for i in range(self.rows):
                if j != i:
                    factor = matrix[i][j]
                    for k in range(self.cols):
                        matrix.matrix[i][k] -= factor * matrix.matrix[j][k]
                        eye.matrix[i][k] -= factor * eye.matrix[j][k]
        return eye

    def padding(self, pad=1, value=0):
        if pad == 0:
            return Matrix(self.matrix)
        bound = [value] * (self.cols + pad * 2)
        m = []
        for _ in range(pad):
            m.append(list(bound))
        for row in self.matrix:
            m.append([value]*pad + row + [value]*pad)
        for _ in range(pad):
            m.append(list(bound))
        return Matrix(m)

    def sub_matrix(self, i, j, row, col):
        if i + row > self.rows or j + col > self.cols:
            raise MatrixError("Sub matrix is out of bounds")
        return Matrix([[self.matrix[x][y] for y in range(j, j + col)] for x in range(i, i + row)])

    def convolve(self, kernel, stride = 1, pad=0, value=0, func = sum):
        if not isinstance(kernel, Matrix):
            raise MatrixError("The kernel should be a Matrix")
        if not isinstance(stride, int):
            raise MatrixError("Stride should be an integer")
        if not callable(func):
            raise MatrixError("Function should be callable")
        stride = max(stride, 1)
        matrix = self.padding(pad, value)
        if kernel.rows > matrix.rows or kernel.cols > matrix.cols:
            raise MatrixError("Kernel is bigger than the matrix")
        output = []
        for i in range(0, matrix.rows - kernel.rows + 1, stride):
            row = []
            for j in range(0, matrix.cols - kernel.cols + 1, stride):
                window = matrix.sub_matrix(i, j, kernel.rows, kernel.cols)
                row.append(func((kernel * window).flatten()))
            output.append(row)
        return Matrix(output)

    def conv_output_size(self, kernel, stride, pad = 0):
        return {"row": (self.rows - kernel.rows + 2 * pad + stride) // stride, "col": (self.cols - kernel.cols + 2 * pad + stride) // stride}

    @staticmethod
    def fill(i, j, value):
        return Matrix([[value for _ in range(j)] for _ in range(i)])

image = Matrix([
    [1, 4, 5],
    [3, -9, 0],
    [1, 20, 0]
])
eye = Matrix.fill(2, 2, 1)
cnn = image.convolve(eye, func = max)
print(cnn)
