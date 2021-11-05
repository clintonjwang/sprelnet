import math
import numpy as np
import torch
nn = torch.nn
F = nn.functional

def load_job_weights(job, step):
	return