# CAVE Functions

Controlled Average, Variance, and Extrema (CAVE) functions are a class of functions that transform input data to have a specified average, variance, minimum, and maximum.

## Using CAVE Functions

Do this to run the code.

### Example

Here's an example on how to use it.

## Custom CAVE Base Functions

It's easy to create new base functions for CAVE applications.
To do so, you must create a class that inherits `CAVEBaseFunction` from `cave_base_functions.py`.
All you need to provide are methods for the function and its first three derivatives.

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