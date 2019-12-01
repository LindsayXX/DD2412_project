import matplotlib.pylab as plt
import numpy as np

val = np.array([1670.4644, 1635.7202, 1618.0466, 1605.477, 1594.524, 1584.2817, 1574.1228, 1564.1912,
                  1554.311, 1544.483, 1534.6455, 1524.906, 1515.3182, 1505.651, 1496.1083, 1486.5826, 1477.177,
                  1467.8752, 1458.3955, 1449.2712, 1440.264, 1431.2175, 1421.6034, 1406.4069, 1397.7786,
                1389.4197, 1381.4788, 1373.6088, 1365.789, 1358.1678])


if __name__ == '__main__':
    train = []
    f = open("log.txt", "r")
    for line in f:
        parts = line.split("[")
        parts2 = parts[1].split("]")
        train.append(float(parts2[0]))
    train = np.array(train)
    plt.plot(range(len(train)), train, label="Train")
    plt.plot(range(len(train)), val, label="Val")
    plt.xlabel("Epoch", fontsize=14)
    plt.ylabel("Loss", fontsize=14)
    plt.legend()
    plt.savefig("loss.png")
    plt.show()
