Implementation of a simple neural net based on the book:
http://neuralnetworksanddeeplearning.com

Code was adapted to python 3.7.

The demo application learns to recognize handwritten digits from MINST dataset (http://yann.lecun.com/exdb/mnist/).

Output is in a format of:
```Epoch {epoch_number}: {number_of_digits_net_was_able_to_recognize} / {total_number_of_digits_in_evaluation_data}```

Example output:

```
Epoch 0: 9441 / 10000
Epoch 1: 9493 / 10000
Epoch 2: 9535 / 10000
Epoch 3: 9571 / 10000
Epoch 4: 9537 / 10000
...
Epoch 28: 9618 / 10000
Epoch 29: 9564 / 10000
```