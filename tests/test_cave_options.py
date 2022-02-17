import os
import sys

import torch

sys.path.append(os.path.dirname(os.getcwd()))
from cave import CAVE


if(__name__ == "__main__"):

	test_cases = [
		([None, None, None, None], "opt_none"), # opt_none
		([0.0, None, None, None], "opt_low"), # opt_low
		([None, 10.0, None, None], "opt_high"), # opt_high
		([0.0, 10.0, None, None], "opt_range"), # opt_range
		([None, None, 0.2, None], "opt_mean"), # opt_mean
		([0.0, None, 0.2, None], "opt_cave"), # opt_cave (mean and low)
		([None, 10.0, 0.2, None], "opt_cave"), # opt_cave (mean and high)
		([0.0, 10.0, 0.2, None], "opt_cave"), # opt_cave (mean and range)
		([None, None, None, 0.1], "opt_var"), # opt_var
		([0.0, None, None, 0.1], "opt_cave"), # opt_cave (var and low)
		([0.0, 10.0, None, 0.1], "opt_cave"), # opt_cave (var and range)
		([None, None, 0.2, 0.1], "opt_moments"), # opt_moments
		([0.0, None, 0.2, 0.1], "opt_cave"), # opt_cave (moments and low)
		([None, 10.0, 0.2, 0.1], "opt_cave"), # opt_cave (moments and high)
		([0.0, 10.0, 0.2, 0.1], "opt_cave"), # everything
	]

	for case, key in test_cases:

		c = CAVE(n_step_gd = 30, n_step_nm = 10, lr_gd = 0.1, lr_nm = 0.1)

		ret_func = c.forward(torch.tensor([1, 2, 3]),
							 low = case[0], high = case[1], mean = case[2], var = case[3])

		assert(ret_func.__name__ == key) # seeing if correct function is returned