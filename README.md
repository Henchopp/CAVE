# CAVE Functions

Controlled Average, Variance, and Extrema (CAVE) functions are a class of functions that transform input data to have a specified average, variance, minimum, and maximum.

## About CAVE

The basis of CAVE is finding a linear transform of the input data prior to feeding it to a nonlinear, range-limited function *f*.
CAVE solves
- *target_mean = mean(f(aX + b))* and
- *target_var = var(f(aX + b))*

for variables *a* and *b* using a combination of gradient descent and Newton's method.

CAVE optimization can be used during training of neural networks.
It has a fully implemented numerical solver as well as the backward method for the input data (i.e. the derivative of gradient descent and Newton's method themselves w.r.t. the input data).
This makes it a more memory-efficient algorithm allowing more CAVE optimization steps to be taken.

## Using CAVE Functions

There are two important steps in using CAVE functions, namely the initialization of the class `CAVE` and the `forward` method parameters.

### Initializing a `CAVE` Instance

There are 8 input arguments for initialization, namely:
- `func` (required): The base CAVE function inherited from `CAVEBaseFunction`
- `n_step_gd` (optional): The number of gradient descent steps to take
- `n_step_nm` (optional): The number of Newton's method steps to take
- `lr_gd` (optional): The learning rate of gradient descent
- `lr_nm` (optional): The learning rate of Newton's method
- `a_init` (optional): The starting value for `a`
- `b_init` (optional): The starting value for `b`
- `output_log` (optional): Logs the output for visualization in MATLAB (see `matlab/plot_loss.m`)

For sparse outputs, we recommend using `Sigmoid` from `cave_base_functions.py`.
Otherwise, feel free to use `Softplus` or code your own (see section below).

### Calling the `forward` Function of `CAVE`

The `CAVE` class is inherited from `torch.nn.Module`, so calling `forward` is done in the same manner as other `torch.nn.Module` classes.
The `forward` method takes 7 arguments:
- `x` (required): The input data as a `torch.Tensor` of any shape
- `low` (optional): The minimum *range* of the output (does not guarantee the minimum *value* of the output is `low`)
- `high` (optional): The maximum *range* of the output (does not guarantee the maximum *value* of the output is `high`)
- `mean` (optional): The desired mean of the output
- `var` (optional): The desired variance of the output (use *biased* variance)
- `sparse` (optional): A boolean indicating whether the output data is expected to be sparse, as it applies a transform to improve the stability of the algorithm (used particularly with large data)
- `dim` (optional): The dimension across which to apply CAVE (similar to `torch.mean` and the like)

### Example

Suppose we have are doing single class classification of MNIST data using a neural network.
Our final output is a one-hot vector of size 10 per input image.
During training, assume we use a batch size of 500.
Our output is therefore 10 x 500.

Each column ideally would have a 1 in the correct bin and 0's everywhere else.
The relevant information for `CAVE` is as follows:
- `low = 0.0` as each entry represents a probability
- `high = 1.0` as each entry represents a probability
- `mean = 0.1` as the mean for the one-hot vector of size 10
- `var = 0.09` as the *biased* variance for the one-hot vector of size 10
- `sparse = True` as we expect a sparse output
- `dim = 0` as we only want to sparsify the columns

This makes `CAVE` a perfect candidate for a final activation layer.
We can code up a `CAVE` activation as follows:
```
import torch.nn as nn
from my_network import MyNet
from cave import CAVE
from cave_base_functions import Sigmoid


class MyCAVENetwork(nn.Module):
	def __init__(self):
		super(MYCAVENetwork, self).__init__()
		self.my_net = MyNet()
		self.cave = CAVE(func = Sigmoid())

	def forward(self, x):
		x = self.my_net(x)
		x = self.cave(x, low = 0.0, high = 1.0, mean = 0.1, var = 0.09, sparse = True, dim = 0)
		return x
```

## Custom CAVE Base Functions

It's easy to create new base functions for CAVE applications.
To do so, you must create a class that inherits `CAVEBaseFunction` from `cave_base_functions.py`.
All you need to provide are methods for the function and its first three derivatives as well as the minimum and maximum values of the function's range.

### Example

Your new base function must override four methods:
- `fx(self, x)`, the function itself
- `dfx(self, x)`, the first derivative
- `d2fx(self, x)`, the second derivative
- `d3fx(self, x)`, the third derivative

You should *not* take into consideration the linear transform as it has been implemented for you.
For example, sigmoid is implemented in `cave_base_functions.py` simply as
```
class Sigmoid(CAVEBaseFunction):

	# Initialization
	def __init__(self):
		super().__init__(low = 0.0, high = 1.0)

	# Function
	def fx(self, x):
		return x.sigmoid()

	# First derivative
	def dfx(self, x):
		sig = x.sigmoid()
		return sig * (1 - sig)

	# Second derivative
	def d2fx(self, x):
		sig = x.sigmoid()
		return sig * (1 - sig) * (1 - 2 * sig)

	# Third derivative
	def d3fx(self, x):
		sig = x.sigmoid()
		return sig * (1 - sig) * (6 * sig * (sig - 1) + 1)
```
in its entirety.
Note that it inherits from CAVEBaseFunction, which will automatically calculate the derivatives with respect to the linear transform.