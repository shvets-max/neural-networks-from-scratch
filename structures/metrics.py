

def evaluate_accuracy(network, X_test, y_test):
    correct = 0
    for xi, yi in zip(X_test, y_test):
        out0 = network[0].activate(xi)
        out1 = network[1].activate(out0)
        out2 = network[2].activate(out1)
        prediction = 1 if out2[0] >= 0.5 else 0
        if prediction == yi:
            correct += 1
    return correct / len(y_test)