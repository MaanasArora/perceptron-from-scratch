from tqdm import tqdm


class Perceptron:
    def __init__(self, weights: float = None, bias: float = None):
        self.weights = weights
        self.bias = bias

    def initialize(self, n: int):
        self.weights = [0] * n
        self.bias = 0

    def fit_epoch(self, X: list, y: list[int], learning_rate: float):
        for x_i, y_i in zip(X, y):
            y_pred = self.predict_one(x_i)

            if y_pred != y_i:
                correction = learning_rate * (2 * y_i - 1)
                self.weights = list(
                    map(lambda w, x_ij: w + correction * x_ij, self.weights, x_i)
                )
                self.bias += correction

    def fit(
        self, X: list, y: list[int], epochs: int = 1000, learning_rate: float = 0.1
    ):
        assert len(X) == len(y)

        if not self.weights:
            self.initialize(len(X[0]))

        for _ in tqdm(range(epochs), total=epochs):
            self.fit_epoch(X, y, learning_rate)

    def predict_one(self, X: list[float]):
        prob = sum(map(lambda w, x_i: w * x_i, self.weights, X)) + self.bias

        return 1 if prob > 0 else 0

    def predict(self, X: list):
        return list(map(self.predict_one, X))
