import inspect
import math
from functools import wraps
from vector import Vector
from matrix import Matrix

class StatisticsError(Exception):
    pass

class ClassMethodMeta(type):
    def __new__(cls, name, bases, namespace):
        for key, value in list(namespace.items()):
            if callable(value) and not key.startswith("__"):
                parameters = [*inspect.signature(value).parameters]
                l = len(parameters)
                if name == "Statistics" and l >= 2 and parameters[1] == "data":
                    value = cls.validate_statistics(value)
                if name == "Probability" and l >= 4 and parameters[1] == "space" and parameters[2] == "set_1" and parameters[3] == "set_2":
                    value = cls.validate_probability(value)
                namespace[key] = classmethod(value)
        return super().__new__(cls, name, bases, namespace)

    @staticmethod
    def validate_statistics(func):
        @wraps(func)
        def wrapper(cls, data, *args, **kwargs):
            if not isinstance(data, (Vector, Matrix)):
                raise StatisticsError(
                    "To compute statistics you should use a Vector or Matrix element"
                )
            return func(cls, data, *args, **kwargs)
        return wrapper

    @staticmethod
    def validate_probability(func):
        @wraps(func)
        def wrapper(cls, space, s1, s2, *args, **kwargs):
            if not isinstance(space, (Vector, Matrix)) or not isinstance(s1, (Vector, Matrix)) or not isinstance(s2, (Vector, Matrix)):
                raise StatisticsError(
                    "To compute statistics you should use a Vector or Matrix element"
                )
            if not set(s1).issubset(set(space)) or not set(s2).issubset(set(space)):
                raise StatisticsError("Set is not included in the sample space")
            return func(cls, space, s1, s2, *args, **kwargs)
        return wrapper

class Statistics(metaclass=ClassMethodMeta):
    def sum(cls, data):
        s = 0
        if (isinstance(data, Vector)):
            return sum(data.vector)
        if (isinstance(data, Matrix)):
            for i in range(data.rows):
                for j in range(data.cols):
                    s += data[i][j]
        return s
    def mean(cls, data):
        return cls.sum(data) / len(data)
    def show_attr(cls):
        return cls.__dict__
    def weighted_mean(cls, data, weight = None):
        if (weight is None or isinstance(weight, (int, float))):
            return cls.mean(data)
        if (isinstance(data, Vector) and isinstance(weight, Vector)):
            if (len(weight) != len(data)):
                raise StatisticsError(
                    "Data and Weight have not the same shape"
                )
            return sum(i * j for i,j in zip(data, weight)) / cls.sum(weight)
        elif (isinstance(data, Matrix) and isinstance(weight, Matrix)):
            if Matrix.check_same_shape(data, weight):
                return sum(k * l for i,j in zip(data, weight) for k, l in zip(i, j)) / cls.sum(weight)
            raise StatisticsError("Data and Weight have not the same shape")
    """
    Ancienne version inneficace
    def median(cls, data):
        if isinstance(data, Vector):
            return (data[(len(data) - 1) // 2] + data[len(data) // 2]) / 2 if len(data) % 2 == 0 else data[len(data) // 2]
        elif isinstance(data, Matrix):
            if data.cols % 2 == 1:
                if data.rows % 2 == 1:
                    return data.matrix[data.rows // 2][data.cols // 2]
                else:
                    return (data.matrix[data.rows // 2][(data.cols - 1) // 2] + data.matrix[data.rows // 2][(data.cols) // 2]) / 2
            else:
                if data.rows % 2 == 1:
                    return data.matrix[data.rows // 2][data.cols // 2]
                else:
                    return (data.matrix[data.rows // 2][(data.cols - 1) // 2] + data.matrix[data.rows // 2][(data.cols) // 2]) / 2"""

    def median(cls, data):
        return cls.percentile(data, 50)

    def frequency(cls, data):
        frequency = {}
        for x in data:
            frequency[x] = frequency.get(x, 0) + 1
        return frequency

    def mode(cls, data):
        frequency = cls.frequency(data)
        m = max(frequency.values())
        return [x for x,v in frequency.items() if v == m] if m != 1 else None

    def extent(cls, data):
        return max(data) - min(data)

    def percentile(cls, data, x, method = "linear"):
        index = len(data) - 1
        y = x / 100 if x <= 100 and x >= 0 else x
        if y > 1 or y < 0:
            raise StatisticsError("Incorrect percentile")
        pos = index * y
        v = sorted(data)
        lower = math.floor(pos)
        upper = min(math.ceil(pos), index)
        rounded = min(math.floor(pos + 0.5), index)
        if index == 0:
            return v[0]
        if (method == "higher"):
            return v[upper]
        elif (method == "lower"):
            return v[lower]
        elif (method == "midpoint"):
            return (v[upper] + v[lower]) / 2
        elif (method == "nearest"):
            return v[rounded]
        return v[lower] + (pos - lower) * (v[upper] - v[lower])

    def quartile(cls, data, x = 2, method = "linear"):
        if x < 1 or x > 3:
            raise StatisticsError("Invalid quartile index: expected 1, 2 or 3")
        return cls.percentile(data, x * 25, method)

    def iqr(cls, data, method = "linear"):
       return cls.quartile(data, 3, method) - cls.quartile(data, 1, method)

    def _get_center(cls, data, method, value):
        m = 0
        if method == "mean":
            m = cls.mean(data)
        elif method == "median":
            m = cls.median(data)
        elif method == "mode":
            m = cls.mode(data)
            if m is None or len(m) >= 2:
                raise StatisticsError("No mode or to many modes found, can't compute")
            m = m[0]
        elif method == "value":
            if isinstance(value, (int, float)):
                m = value
            else:
                raise StatisticsError("Value has to be an integer or a float")
        else:
            raise StatisticsError("Incorrect method please enter a correct method")
        return m

    def mean_absolute_deviation(cls, data, method = "mean", value = None):
        m = cls._get_center(data, method, value)
        return sum(abs(xi - m) for xi in data) / len(data)

    def median_absolute_deviation(cls, data, method = "median", value = None):
        m = cls._get_center(data, method, value)
        return cls.median(Vector([abs(xi - m) for xi in data]))

    def outliers_bounds(cls, data, tukey_factor=1.5):
        q1 = cls.quartile(data, 1)
        q3 = cls.quartile(data, 3)
        iqr = q3 - q1
        return [q1 - tukey_factor * iqr, q3 + tukey_factor * iqr]

    def squared_deviation(cls, data, ddof = 0, center="mean", value = None):
        m = cls._get_center(data, center, value)
        return sum((xi - m) ** 2 for xi in data) / max(len(data) - ddof, 1)

    def var(cls, data, ddof = 0):
        return cls.squared_deviation(data, ddof, "mean", None)

    def standard_deviation(cls, data, ddof = 0):
        return math.sqrt(cls.var(data, ddof))

    def covariance(cls, d_x, d_y, ddof = 0, center = "mean", value=None):
        mx = cls._get_center(d_x, center, value)
        my = cls._get_center(d_y, center, value)
        if len(d_x) != len(d_y):
            raise StatisticsError("Should have the same length")
        return sum((xi - mx) * (yi - my) for xi, yi in zip(d_x, d_y)) / max(len(d_x) - ddof, 1)

    def coefficient_of_variation(cls, data):
        return cls.standard_deviation(data) / cls.mean(data)

    def pearson_correlation(cls, d_x, d_y):
        return cls.covariance(d_x, d_y) / (cls.standard_deviation(d_x) * cls.standard_deviation(d_y))

    def covariance_array(cls, d_n):
        if not isinstance(d_n, list):
            raise StatisticsError("Not an array")
        for i in d_n:
            if not isinstance(i, (Vector, Matrix)):
                raise StatisticsError("Should be Matrix or Vector")
        m = []
        for i in range(len(d_n)):
            m.append([])
            for j in range(len(d_n)):
                if len(d_n[i]) != len(d_n[j]):
                    raise StatisticsError("Should have same length")
                m[i].append(cls.covariance(d_n[i], d_n[j]))
        return Matrix(m)

    def corr_array(cls, d_n):
        cov_arr = cls.covariance_array(d_n)
        return Matrix([cov_arr[i][j] / cls.var(d_n[i][j]) for j in range(d_n.cols)] for i in range(d_n.rows))

    def z_score(cls, data, x):
        if not isinstance(x, (int, float)):
            raise StatisticsError("X has to be an integer or float")
        std = cls.standard_deviation(data)
        if Matrix.is_close(std, 0):
            raise StatisticsError("Standard deviation is close to zero, can't compute z-score")
        return (x - cls.mean(data)) / std

    def standard_scaler(cls, data):
        std = cls.standard_deviation(data)
        if Matrix.is_close(std, 0):
            raise StatisticsError("Standard deviation is close to zero, can't compute z-score")
        mean = cls.mean(data)
        return Vector([(x - mean) / std for x in data])

    def robust_scaler(cls, data):
        q3 = cls.quartile(data, 3)
        q1 = cls.quartile(data, 1)
        q2 = cls.quartile(data, 2)
        iqr = q3 - q1
        return Vector([(x - q2) / iqr for x in data])

    def mean_scaler(cls, data):
        mean = cls.mean(data)
        return Vector([x - mean for x in data])

    def slope(cls, d_x, d_y):
        return cls.covariance(d_x, d_y) / cls.var(d_x)

    def linear_regression(cls, d_x, d_y):
        slope = cls.slope(d_x, d_y)
        intercept = cls.mean(d_y) - slope * cls.mean(d_x)
        return slope, intercept, lambda x: intercept + slope * x

    def linear_regression_predict(cls, d_x, d_y):
        regression = cls.linear_regression(d_x, d_y)
        return Vector([regression[2](x) for x in d_x])

    def linear_regression_error(cls, predict, d_y, method="sub"):
        if method == "square":
            return Vector([(yp - y)**2 for y, yp in zip(d_y, predict)])
        return Vector([yp - y for y, yp in zip(d_y, predict)])

    def mean_squared_error(cls, predict, d_y):
        return sum((yp - y)**2 for y, yp in zip(d_y, predict)) / len(d_y)

    def root_mean_squared_error(cls, predict, d_y):
        return math.sqrt(cls.mean_squared_error(predict, d_y))

    def r2_score(cls, predict, d_y):
        ssres = sum(cls.linear_regression_error(predict, d_y, "square"))
        mean = cls.mean(d_y)
        sstot = sum((x - mean)**2 for x in d_y)
        return 1 - (ssres / sstot)

    def min_max_scaler(cls, data):
        extent = cls.extent(data)
        mi = min(data)
        return Vector([(x - mi) / extent for x in data])

    def sigmoid(cls, x):
        return 1 / (1 + math.exp(-x))

    def logistic_regression(cls, d_x, d_y):
        regression = cls.linear_regression(d_x, d_y)
        return regression[0], regression[1], cls.sigmoid
    def logistic_regression_predict(cls, d_x, d_y):
        predict = cls.linear_regression_predict(d_x, d_y)
        return Vector([predict[2](x) for x in d_x])
    def matrix_softmax(cls, d_x, axis = 0):
        if not isinstance(d_x, Matrix):
            raise StatisticsError("Dataset have to be a matrix")
        matrix = d_x.transpose() if axis == 1 else d_x
        return Matrix([cls.softmax(matrix[i]).vector for i in range(matrix.rows)])
    def matrix_log_softmax(cls, d_x, axis = 0):
        if not isinstance(d_x, Matrix):
            raise StatisticsError("Dataset have to be a matrix")
        matrix = d_x.transpose() if axis == 1 else d_x
        return Matrix([cls.log_softmax(matrix[i]).vector for i in range(matrix.rows)])
    def softmax(cls, d_x):
        x_max = max(d_x)
        v_x = Vector([math.exp(x - x_max) for x in d_x])
        s = sum(v_x)
        return Vector([x / s for x in v_x])
    def log_softmax(cls, d_x):
        v = cls.softmax(d_x)
        return Vector([math.log(x) for x in v])
    def least_squares(cls, d_x, d_y):
        if not isinstance(d_x, Matrix):
            raise StatisticsError("Features have to be a Matrix")
        t_x = Matrix.transpose(d_x)
        return ((t_x @ d_x).gauss_jordan()) @ t_x @ d_y
    def matrix_linear_regression(cls, d_x, d_y):
        coeff = cls.least_squares(d_x, d_y)
        def calc_intercept():
            mean_y = cls.mean(d_y)
            mean_x = []
            dx = d_x.transpose()
            for x in dx:
                mean_x.append(cls.mean(x))
            return {"intercept": mean_y - sum(coeff[i] * mean_x[i] for i in range(len(dx))), "mean_y": mean_y, "mean_x": mean_x}
        def predict(d_x):
            v = []
            for i in range(d_x.rows):
                v.append(intercept + sum(ai * xi for ai, xi in zip(coeff, d_x[i])))
            return Vector(v)
        fit = calc_intercept()
        intercept = fit["intercept"]
        return {"coeff": coeff, "intercept": intercept, "predict": predict, "mean_y": fit["mean_y"], "mean_x": fit["mean_x"]}

class Probability(metaclass=ClassMethodMeta):
    def probability(cls, data, x):
        frequency = Statistics.frequency(data)
        if not isinstance(x, (int, float)):
            raise StatisticsError("X should be an integer or float")
        try:
            return frequency[x]/len(data)
        except Exception:
            return 0
    def complement(cls, data, x):
        return 1 - cls.probability(data, x)
    def intersection(cls, space, set_1, set_2):
        s1 = set(set_1)
        s2 = set(set_2)
        return len(s1 & s2) / len(space)
    def union(cls, space, set_1, set_2):
        nb = len(set_1) + len(set_2)
        return (nb/len(space) - cls.intersection(space, set_1, set_2))
    def conditional(cls, space, set_1, set_2):
        return cls.intersection(space, set_1, set_2) / (len(set(set_2)) / len(space))
    def is_independent(cls, space, set_1, set_2):
        p2 = len(set(set_2))
        p1 = len(set(set_1))
        p = len(space)
        p1 /= p
        p2 /= p
        return Matrix.is_close(cls.intersection(space, set_1, set_2), (p1 * p2))
    def bayes(cls, space, set_1, set_2):
        p1 = len(set_1) / len(space)
        p2 = len(set_2) / len(space)
        return cls.conditional(space, set_2, set_1) * p1 / p2
    class Distribution(metaclass=ClassMethodMeta):
        def bernoulli(cls, p, x):
            if not isinstance(p, (int, float)) or not isinstance(x, (int, float)) or x not in (0,1) or p < 0:
                raise StatisticsError("P and X should be positive numbers")
            return p**x*(1 - p)**(1 - x)
        def binomial(cls, p, n, k):
            if not isinstance(p, (int, float)) or not isinstance(k, int) or not isinstance(n, int):
                raise StatisticsError("P and X should be positive numbers")
            if k < 0 or k > n:
                raise StatisticsError("k is not in the range")
            return p**k*(1 - p)**(n - k) * cls.combination(n, k)
        def factorial(cls, x):
            if x < 0:
                raise StatisticsError("The factorial is not defined for negative numbers")
            elif x <= 1:
                return 1
            return x * cls.factorial(x - 1)
        def arrange(cls, n, k):
            return cls.factorial(n) / cls.factorial(n - k)
        def combination(cls, n, k):
            return cls.arrange(n, k) / cls.factorial(k)
        def esperance(cls, v, p):
            if not isinstance(v, Vector) or not isinstance(p, Vector):
                raise StatisticsError("Values and Probabilities have to be of type Vector")
            elif len(v) != len(p):
                raise StatisticsError("The length of values is not equal to the length of the probabilities")
            return sum(x * y for x,y in zip(v, p))
        def variance(cls, v, p):
            ex = cls.esperance(v, p)
            ex2 = sum((x ** 2)*y for x,y in zip(v, p))
            return ex2 - ex ** 2
        def uniform(cls, a, b):
            c = b - a
            if Matrix.is_close(c, 0):
                raise StatisticsError("Value close to 0")
            return 1 / c
