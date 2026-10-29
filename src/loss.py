import inspect
from functools import wraps
import math
from vector import Vector
from matrix import Matrix
from statistics import Statistics

class LossValidationError(Exception):
    pass

class LossComputationError(Exception):
    pass

class MetaLoss(type):
    def __new__(cls, name, bases, namespace):
        for key, value in list(namespace.items()):
            if callable(value) and not key.startswith('_'):
                namespace[key] = cls.validate_parameters(classmethod(value))

        return super().__new__(cls, name, bases, namespace)

    @staticmethod
    def validate_parameters(func):
        @wraps(func)
        def wrapper(cls, *args, **kwargs):
            signature = inspect.signature(func)
            params = [*signature.parameters]
            values = signature.bind(cls, *args, **kwargs).arguments
            len_params = len(params)
            if len_params >= 3 and params[1] == 'y_true' and params[2] == 'y_pred':
                y_true = values[params[1]]
                y_pred = values[params[2]]
                if not isinstance(y_true, Vector) or not isinstance(y_pred, Vector):
                    raise LossValidationError("y_pred and y_true must be of type Vector")
                if len(y_true) != len(y_pred):
                    raise LossValidationError("y_pred and y_true must have the same length")
                if y_true.empty():
                    raise LossValidationError("y_true and y_pred must contain at least one element")
            if len_params >= 3 and params[1] == 'class_indices' and params[2] == 'class_probabilities':
                class_indices = values[params[1]]
                class_probabilities = values[params[2]]
                if not isinstance(class_indices, Vector) or not isinstance(class_probabilities, Matrix):
                    raise LossValidationError("class_indices must be of type Vector and  class_probabilities must be of type Matrix")
                if len(class_indices) != class_probabilities.rows:
                    raise LossValidationError("the length of class_indices must be equal to the number of cols of class_probabilities")
                if class_indices.empty():
                    raise LossValidationError("class_indices must not be empty")
            return func(cls, *args, **kwargs)
        return wrapper

class Loss(metaclass=MetaLoss):
    @staticmethod
    def _epsilon(x: int | float):
        if not isinstance(x, (int, float)):
            raise LossComputationError("X must be of type int or float")
        return abs(x) < 1e-12

    def loss(cls, y_true, y_pred):
        return Vector([y_true[i] - y_pred[i] for i in range(len(y_pred))])

    def mean_squared_error(cls, y_true, y_pred):
        data_len = len(y_true)
        return sum((y_true[i] - y_pred[i]) ** 2 for i in range(data_len)) / data_len

    def mse(cls, y_true, y_pred):
        return cls.mean_squared_error(y_true, y_pred)

    def mean_absolute_error(cls, y_true, y_pred):
        data_len = len(y_true)
        return sum(abs(y_true[i] - y_pred[i]) for i in range(data_len)) / data_len

    def mae(cls, y_true, y_pred):
        return cls.mean_absolute_error(y_true, y_pred)

    @staticmethod
    def _huber(y_true, y_pred, delta):
        e = (y_true - y_pred)
        return 1 / 2 * e ** 2 if abs(e) <= delta else delta * (abs(e) - delta * 1 / 2)

    def huber(cls, y_true, y_pred, delta = 1, method = "none"):
        result = Vector([cls._huber(y_true[i], y_pred[i], delta) for i in range(len(y_true))])
        if method == 'none':
            return result
        elif method == 'mean':
            return Statistics.mean(result)
        elif method == 'sum':
            return sum(result)
        raise LossComputationError("aggregation method must be : mean / none / sum")

    def logcosh(cls, y_true, y_pred):
        len_data = len(y_pred)
        return sum(math.log(math.cosh(y_true[i] - y_pred[i])) for i in range(len_data)) / len_data

    def mean_absolute_percentage_error(cls, y_true, y_pred):
        len_data = len(y_true)
        return (100 / len_data) * sum(abs((y_true[i] - y_pred[i]) / y_true[i]) for i in range(len_data))

    def mape(cls, y_true, y_pred):
        return cls.mean_absolute_percentage_error(y_true, y_pred)

    def mean_squared_logarithmic_error(cls, y_true, y_pred):
        len_data = len(y_pred)
        return sum((math.log(1 + y_true[i]) - math.log(1 + y_pred[i])) ** 2 for i in range(len_data)) / len_data

    def msle(cls, y_true, y_pred):
        return cls.mean_squared_logarithmic_error(y_true, y_pred)

    def root_mean_squared_error(cls, y_true, y_pred):
        return math.sqrt(cls.mse(y_true, y_pred))

    def rmse(cls, y_true, y_pred):
        return cls.root_mean_squared_error(y_true, y_pred)

    def mean_squarred_percentage_error(cls, y_true, y_pred):
        len_data = len(y_true)
        return (1 / len_data) * sum(((y_true[i] - y_pred[i]) / y_true[i]) ** 2 for i in range(len_data))

    def mspe(cls, y_true, y_pred):
        return cls.mean_squarred_percentage_error(y_true, y_pred)

    def poisson(cls, y_true, y_pred):
        len_data = len(y_true)
        return sum(y_pred[i] - y_true[i] * math.log(y_pred[i]) for i in range(len_data)) * (1 / len_data)

    @staticmethod
    def _pinball(y_true, y_pred, q):
        e = y_true - y_pred
        return q * e if e >= 0 else (q - 1) * e

    def pinball(cls, y_true, y_pred, q = 1, method = "none"):
        if q < 0 or q > 1:
            raise LossComputationError("q must be between 0 and 1")
        result = Vector([cls._pinball(y_true[i], y_pred[i], q) for i in range(len(y_true))])
        if method == 'none':
            return result
        elif method == 'mean':
            return Statistics.mean(result)
        elif method == 'sum':
            return sum(result)
        raise LossComputationError("aggregation method must be : mean / none / sum")

    def weighted_mse(cls, y_true, y_pred, weight, method = "mean"):
        loss = cls.loss(y_true, y_pred)
        len_data = len(y_true)
        if len(weight) != len_data:
            raise LossComputationError("Weights and Data must have the same length")
        if method == 'mean':
            return (1 / len_data) * sum(weight[i] * loss[i] ** 2 for i in range(len_data))
        if method == 'weighted_mean':
            return (1 / sum(weight)) * sum(weight[i] * loss[i] ** 2 for i in range(len_data))
        raise LossComputationError("Unknow method")

    def binary_cross_entropy(cls, y_true, y_pred):
        len_data = len(y_true)
        return - 1 / len_data * sum(y_true[i] * math.log(y_pred[i]) + (1 - y_true[i]) * math.log(1 - y_pred[i]) for i in range(len_data))

    def bce(cls, y_true, y_pred):
        return cls.binary_cross_entropy(y_true, y_pred)

    def categorical_cross_entropy(cls, y_true, y_pred):
        len_data = len(y_true)
        return - sum(y_true[i] * math.log(y_pred[i]) for i in range(len_data))

    def cce(cls, y_true, y_pred):
        return cls.categorical_cross_entropy(y_true, y_pred)

    @staticmethod
    def _hinge(y_true, y_pred):
        if y_true < -1 or y_true > 1:
            raise LossValidationError("y_true must be between -1 and 1")
        return max(0, 1 - y_true * y_pred)

    def hinge(cls, y_true, y_pred, method = 'none'):
        len_data = len(y_true)
        result = Vector([cls._hinge(y_true[i], y_pred[i]) for i in range(len_data)])
        if method == 'none':
            return result
        if method == 'mean':
            return Statistics.mean(result)
        if method == 'sum':
            return sum(result)
        raise LossComputationError("aggregation method must be : mean / none / sum")

    def squared_hinge(cls, y_true, y_pred, method = "none"):
        len_data = len(y_true)
        result = Vector([cls._hinge(y_true[i], y_pred[i]) ** 2 for i in range(len_data)])
        if method == 'none':
            return result
        if method == 'mean':
            return Statistics.mean(result)
        if method == 'sum':
            return sum(result)
        raise LossComputationError("aggregation method must be : mean / none / sum")

    def sparse_categorical_cross_entropy(cls, class_indices, class_probabilities):
        len_data = len(class_indices)
        result = 0
        for i in range(len_data):
            if not math.isclose(sum(class_probabilities[i]), 1):
                raise LossValidationError("Sum of all probabilities must be equal to 1")
            y_true = class_indices[i]
            y_pred = 0
            if not isinstance(y_true, int) or y_true < 0 or isinstance(y_true, bool):
                raise LossValidationError(f"y_true must be a positive integer between 0 and {class_probabilities.cols - 1}")
            try:
                y_pred = class_probabilities[i][y_true]
            except IndexError:
                raise LossValidationError("Incorrect class index")
            if y_pred <= 0 or y_pred > 1 or not math.isfinite(y_pred):
                raise LossComputationError("y_pred must be between 0 and 1")
            result += math.log(y_pred)
        return - (1 / len_data) * result

    def focal_loss(cls, y_true, y_pred, alpha=0.25, gamma=2.0):
        if not isinstance(alpha, (int, float)) or isinstance(alpha, bool) or not math.isfinite(alpha) or not 0 <= alpha <= 1:
            raise LossValidationError("alpha must be between 0 and 1")

        if not isinstance(gamma, (int, float)) or isinstance(gamma, bool) or not math.isfinite(gamma) or gamma < 0:
            raise LossValidationError("gamma must be a finite number greater than or equal to 0")

        len_data = len(y_true)
        result = 0.0

        for i in range(len_data):
            y = y_true[i]
            p = y_pred[i]

            if y not in (0, 1) or isinstance(y, bool):
                raise LossValidationError("y_true must contain only 0 or 1")

            if not isinstance(p, (int, float)) or isinstance(p, bool) or not math.isfinite(p) or not 0 <= p <= 1:
                raise LossValidationError("y_pred must contain probabilities between 0 and 1")

            p_t = p if y == 1 else 1 - p
            alpha_t = alpha if y == 1 else 1 - alpha
            p_t = max(p_t, 1e-12)
            result += -alpha_t * (1 - p_t) ** gamma * math.log(p_t)
        return result / len_data

    def brier_score(cls, class_indices, class_probabilities):
        len_data = len(class_indices)
        result = 0.0
        num_classes = class_probabilities.cols
        for i in range(len_data):
            y_true = class_indices[i]
            probabilities = class_probabilities[i]
            if not isinstance(y_true, int) or isinstance(y_true, bool) or not 0 <= y_true < num_classes:
                raise LossValidationError(f"Each class index must be an integer between 0 and {num_classes - 1}")
            if not math.isclose(sum(probabilities), 1.0, rel_tol=1e-9, abs_tol=1e-9):
                raise LossValidationError("Sum of all probabilities must be equal to 1")
            sample_loss = 0.0
            for c in range(num_classes):
                p = probabilities[c]
                if not isinstance(p, (int, float)) or isinstance(p, bool) or not math.isfinite(p) or not 0 <= p <= 1:
                    raise LossValidationError("Probabilities must be finite numbers between 0 and 1")
                y = 1.0 if c == y_true else 0.0
                sample_loss += (p - y) ** 2
            result += sample_loss
        return result / len_data

    def kl_divergence(cls, y_true, y_pred):
        result = 0
        if not math.isclose(sum(y_true), 1) or not math.isclose(sum(y_pred), 1):
            raise LossValidationError("The sum of the probabilities of y_true and y_pred must be equal to 1")
        for p, q in zip(y_true, y_pred):
            if p < 0 or p > 1 or q <= 0 or q > 1:
                raise LossValidationError("Probability is out of bounds")
            if p == 0:
                continue
            result += p * math.log(p / q)
        return result

    def exponential(cls, y_true, y_pred):
        len_data = len(y_true)
        result = 0
        for i, j in zip(y_true, y_pred):
            if (i != -1 and i != 1) or j < -1 or j > 1:
                raise LossValidationError("Scores must be between -1 and 1")
            result += math.exp(-i * j)
        result *= (1 / len_data)
        return result

    def contrastive(cls, y_true, y_pred, label, margin = 1):
        if isinstance(label, bool) or label < 0 or label > 1:
            raise LossValidationError("label must be between 0 and 1")
        if not math.isfinite(margin):
            raise LossValidationError("margin must be an finite number")
        d = math.sqrt(sum((i - j) ** 2 for i, j in zip(y_true, y_pred)))
        return label * d ** 2 + (1 - label) * max(0, margin - d) ** 2

    @classmethod
    def triplet(cls, anchor, positive, negative, margin=1.0):
        vectors = (anchor, positive, negative)

        if not all(isinstance(v, Vector) for v in vectors):
            raise LossValidationError("anchor, positive and negative must be Vectors")

        if not len(anchor) or len(anchor) != len(positive) or len(anchor) != len(negative):
            raise LossValidationError("All vectors must have the same non-zero length")

        if isinstance(margin, bool) or not isinstance(margin, (int, float)):
            raise LossValidationError("margin must be a finite positive number")

        if not math.isfinite(margin) or margin <= 0:
            raise LossValidationError("margin must be a finite positive number")

        d_positive = math.sqrt(sum((a - p) ** 2 for a, p in zip(anchor, positive)))
        d_negative = math.sqrt(sum((a - n) ** 2 for a, n in zip(anchor, negative)))

        return max(0.0, d_positive - d_negative + margin)

    @classmethod
    def cosine_embedding(cls, y_true, y_pred, label, margin=0.0):
        if not isinstance(y_true, Vector) or not isinstance(y_pred, Vector):
            raise LossValidationError("y_true and y_pred must be Vectors")

        if len(y_true) == 0 or len(y_true) != len(y_pred):
            raise LossValidationError("Vectors must have the same non-zero length")

        if isinstance(label, bool) or not isinstance(label, int) or label not in (-1, 1):
            raise LossValidationError("label must be -1 or 1")

        if isinstance(margin, bool) or not isinstance(margin, (int, float)):
            raise LossValidationError("margin must be a finite number")

        if not math.isfinite(margin) or not -1 <= margin <= 1:
            raise LossValidationError("margin must be between -1 and 1")

        dot = sum(a * b for a, b in zip(y_true, y_pred))
        norm_true = math.sqrt(sum(a * a for a in y_true))
        norm_pred = math.sqrt(sum(b * b for b in y_pred))

        if norm_true == 0 or norm_pred == 0:
            raise LossValidationError("Cosine similarity is undefined for zero vectors")

        cosine = dot / (norm_true * norm_pred)
        cosine = max(-1.0, min(1.0, cosine))

        if label == 1:
            return 1.0 - cosine

        return max(0.0, cosine - margin)

    @classmethod
    def dice(cls, y_true, y_pred, epsilon=1e-7):
        if not isinstance(y_true, Vector) or not isinstance(y_pred, Vector):
            raise LossValidationError("y_true and y_pred must be Vectors")

        if len(y_true) == 0 or len(y_true) != len(y_pred):
            raise LossValidationError("Vectors must have the same non-zero length")

        if isinstance(epsilon, bool) or not isinstance(epsilon, (int, float)):
            raise LossValidationError("epsilon must be a finite positive number")

        if not math.isfinite(epsilon) or epsilon <= 0:
            raise LossValidationError("epsilon must be a finite positive number")

        for value in y_true:
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise LossValidationError("Targets must be finite numbers between 0 and 1")
            if not math.isfinite(value) or not 0 <= value <= 1:
                raise LossValidationError("Targets must be finite numbers between 0 and 1")

        for value in y_pred:
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise LossValidationError("Predictions must be finite numbers between 0 and 1")
            if not math.isfinite(value) or not 0 <= value <= 1:
                raise LossValidationError("Predictions must be finite numbers between 0 and 1")

        intersection = sum(p * y for p, y in zip(y_pred, y_true))
        total = sum(y_pred) + sum(y_true)

        return 1.0 - (2.0 * intersection + epsilon) / (total + epsilon)

    @classmethod
    def wasserstein(cls, y_true, y_pred):
        if not isinstance(y_true, Vector) or not isinstance(y_pred, Vector):
            raise LossValidationError("y_true and y_pred must be Vectors")

        if len(y_true) == 0 or len(y_pred) == 0:
            raise LossValidationError("Vectors must be non-empty")

        for value in y_true:
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
                raise LossValidationError("Critic scores must be finite numbers")

        for value in y_pred:
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
                raise LossValidationError("Critic scores must be finite numbers")

        return sum(y_true) / len(y_true) - sum(y_pred) / len(y_pred)

class Regularization:
    pass
