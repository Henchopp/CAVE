# CAVE Functions

Controlled Average, Variance, and Extrema (CAVE) functions are a class of functions that transform input data to have a specified average, variance, minimum, and/or maximum.

## About CAVE

The basis of CAVE is finding a linear transform of the input data `X` prior to feeding it to a range-limited activation function `f`.
The output `Y` has the following characteristics
- `Y.amin() >= low`
- `Y.amax() <= high`
- `(Y.mean() - mean).abs() < eps`
- `(Y.var() - var).abs() < eps`
for some `eps` where `[low, high, mean, var]` are user-defined.

The CAVE function has the following form for different combinations:
```
if low and high:
    CAVE(X) = (high - low) / (f.max - f.min) * (f(a * X + b) - f.min) + low

elif f.low and low:
    CAVE(X) = f(a * X + b) - f.low + low

elif f.low and high:
    CAVE(X) = -f(a * X + b) + f.low + high

elif f.high and low:
    CAVE(X) = -f(a * X + b) + f.high + low

elif f.high and high:
    CAVE(X) = f(a * X + b) - f.high + high
```
where `a` and `b` are found to minimize the mean-squared error between the mean and/or variance.
Both `a` and `b` are found via gradient descent and Newton's second order optimization.

CAVE optimization can be used during training of neural networks since gradient descent and Newton's method optimization of `a` and `b` are themselves differentiable with respect to `X`.
Efficient implementations of the backwards method results in faster calculations and with fewer memory requirements.

## Using CAVE Functions

The important steps in using CAVE are the initialization of the class `CAVE` and its `forward` function.
Class `CAVE` inherits from `torch.nn.Module` and can be used as a regular activation function.

### Initializing a `CAVE` Instance

There are 13 input arguments for initializing `CAVE` that are all optional.
At least three of `low`, `high`, `mean`, and `var` must be specified however.
The arguments are:
- `low` (`float` or `torch.Tensor`): The lower bound of the desired output range. If no desired lower bound, specify `None`. Default: `None`.
- `high` (`float` or `torch.Tensor`): The upper bound of the desired output range. If no desired upper bound, specify `None`. Default: `None`.
- `mean` (`float` or `torch.Tensor`): The mean of the desired output. If no desired mean, specify `None`. Default: `None`.
- `var` (`float` or `torch.Tensor`): The variance of the desired output. If no desired variance, specify `None`. Default: `None`.
- `func` (`CAVEBaseFunction`): The base CAVE function. Default when `low` xor `high` is specified is `Softplus`. Default when `low` and `high` is specified is `Sigmoid`.
- `dim` (`None`, `int`, or list of `int`): The dimension(s) to apply CAVE over. `None` applies CAVE over all dimensions. Default: `None`.
- `unbiased` (`bool`): Indicates if the variance is calculated with Bessel's correction (`True`) or without `False`. Default: `True`.
- `n_step_gd` (`int`): The number of gradient descent steps to take. Default: `5`.
- `n_step_nm` (`int`): The number of Newton's method steps to take. Default: `10`.
- `lr_gd` (`float`): The learning rate of gradient descent. Default: `1.0`.
- `lr_nm` (`float`): The learning rate of Newton's method. Default: `1.0`.
- `a_init` (`float` or `torch.Tensor`): The starting value for `a`. Default: `1.0`.
- `b_init` (`float` or `torch.Tensor`): The starting value for `b`. Default: `0.0`.

### Calling the `forward` Function of `CAVE`

The `CAVE` class is inherited from `torch.nn.Module`, so calling `forward` is done in the same manner as other `torch.nn.Module` classes.
The `forward` method takes 1 required argument:
- `x` (`torch.Tensor`): The input data.

The optional arguments are any of the 13 `__init__` arguments.

### Example: Classification

In single label classification, the output of a neural network is typically a one-hot vector `pred` of size `N x C` where `N` is the batch size and `C` is the number of classes.
The mean value should be `1 / N` for each sample, `low = 0.0` for all, and `high = 1.0` for all.
We can initialize `CAVE` as follows:
```
cave = CAVE(low = 0.0,
            high = 1.0,
            mean = 1.0 / N,
            dim = 1)
```
We call its `forward` method by
```
output = cave(pred)
```
to get vectors with classification probabilities.

### Example: Mean and Variance Matching

Suppose we are training a neural network that restores RGB images where the mean and variance of each image are known.
During training, we would have a batch `ims` of size `N x 3 x H x W`, the corresponding mean values `means` of size `N x 1 x 1 x 1`, and the corresponding variance values `vars` of size `N x 1 x 1 x 1`.
We can create a `CAVE` instance by coding the following:
```
cave = CAVE(low = 0.0,
            high = 1.0,
            dim = [1,2,3])
```
We can then use the `forward` method by
```
means = ims.mean(dim = [1,2,3], keepdim = True)
vars = ims.var(dim = [1,2,3], keepdim = True)
output = cave(ims, mean = means, var = vars, dim = [1,2,3])
```
where the following would evaluate to `True`:
- `output.amin() >= low`
- `output.amax() <= high`
- `((output.mean(dim = [1,2,3], keepdim = True) - means).abs() < eps).all()`
- `((Y.var(dim = [1,2,3], keepdim = True) - vars).abs() < eps).all()`
for some small `eps`.

## Custom CAVE Base Functions

It's easy to create new base functions for CAVE applications.
To do so, you must create a class that inherits `CAVEBaseFunction` from `cave_base_functions.py`.
All you need to provide are methods for the function and its first three derivatives as well as the minimum and/or maximum values of the function's range.

To work properly, a `CAVEBaseFunction` must have the following properties:
- It operates element-wise
- Its range is semi-infinite or finite
- It is strictly increasing (or decreasing)
- It is smooth and continuous

### Example

Your new base function must specify at least one of `low` and `high`.
Additionally, you must code these four methods:
- `fx(self, x)`, the function itself
- `dfx(self, x)`, the first derivative
- `d2fx(self, x)`, the second derivative
- `d3fx(self, x)`, the third derivative

Keep in mind that these four functions should operate element-wise on any size `torch.Tensor`.
You should *not* take into consideration the linear transform as it has been implemented for you.
For example, `Sigmoid` is implemented simply as
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
Note that it inherits from `CAVEBaseFunction`, which automatically finds the `forward` and `backward` calculations needed to use CAVE.