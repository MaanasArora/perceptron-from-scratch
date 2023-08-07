from perceptron import Perceptron


if __name__ == "__main__":
    perceptron = Perceptron()

    X = [[1, 1], [1, 0], [0, 1], [0, 0]]
    y = [1, 1, 1, 0]

    perceptron.fit(X, y, epochs=200, learning_rate=0.1)
    pred = perceptron.predict(X)

    for x_i, y_i, y_pred in zip(X, y, pred):
        print(f"{x_i} -> {y_pred} (expected {y_i})")
