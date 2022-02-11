import torch
from utils.cave_base_functions import Sigmoid, Softplus
from utils.gradient_step_a import GSA
from utils.gradient_step_b import GSB
from utils.newton_step_a import NSA
from utils.newton_step_b import NSB



class CAVE(torch.nn.Module):
	"""docstring for CAVE"""

	def __init__(self, n_step_gd, n_step_nm, lr_gd, lr_nm,
	             adam = False, a_init = 1.0, b_init = 0.0):
		super(CAVE, self).__init__()

		self.softplus = Softplus()
		self.sigmoid = Sigmoid()


	def opt_none(self, x):
		return x

	def opt_low(self, x, low):
		return self.softplus.fx(x) + low

	def opt_high(self, x, high):
		return -self.softplus.fx(x) + high

	def opt_mean(self, x, mean):
		return x - torch.mean(x) + mean

	def opt_var(self, x, var):
		return torch.sqrt(var / torch.var(x)) * x

	def opt_range(self, x, low, high):
		return (high - low) * self.sigmoid.fx(x) + low

	def opt_moments(self, x, mean, var):
		return torch.sqrt(var / torch.var(x)) * (x - torch.mean(x)) + mean

	def opt_all(self, x, low, high, mean, var):
		return

	def forward(self, x, low = None, high = None, mean = None, var = None):
		return


'''
INPUT VARIABLES for CAVE

forward
-	low = None
-	high = None
-	mean = None
-	var = None

__init__
-	a_init = 1.0
-	b_init = 0.0
-	n_step_gd
-	n_step_nm
-	lr_gd
-	lr_nm
-	adam = False


USAGE (CAVE)
cave = CAVE(init params)
data = torch.rand(100)
mean = torch.Tensor([0.1])
var = torch.Tensor([0.05])
low = torch.zeros(1)
high = torch.ones(1)
output = cave(...) # arguments are forward method


USAGE (opt_all: standard gradient descent)
gsa = GSA()
gsb = GSB()
nsa = NSA()
nsb = NSB()
data = torch.rand(100)
mean = 0.1
var = 0.05
low = 0.0
high = 1.0

# See files for argument usage
gsa_out = gsa(...)
gsb_out = gsb(...)
nsa_out = nsa(...)
nsb_out = nsb(...)

if gradient descent:
	a -= lr_gd * gsa_out
	b -= lr_gd * gsb_out

elif newton's method:
	a -= lr_nm * nsa_out
	b -= lr_nm * nsb_out
'''

c = CAVE(10, 2, 0.1, 0.2)

x = torch.Tensor([1, 2, 3, 4, 5, 6, 7, 8])

print(c.opt_range(x, -1, 1))
###