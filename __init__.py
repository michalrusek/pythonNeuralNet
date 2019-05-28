import numpy as np
import network2
import minst_loader

def main():
    training_data, validation_data, test_data = \
        minst_loader.load_data_wrapper()
    net = network2.Network([784, 30, 10], cost=network2.CrossEntropyCost)
    net.SGD(training_data, 30, 10, 0.5, lmbda=5.0, evaluation_data=validation_data)
    pass

main()