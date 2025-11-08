import os, sys
import math

codebase = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
sys.path.append(codebase)

import compute_privacy_sgm

training_sample_size = 480 * 200 # 50000-2000 | 480 * 200
batch_size = 64
noise_multiplider = 1.25
epochs = 100
delta = 1.0 / training_sample_size
# delta = 1e-5
n = training_sample_size

print('eps=', compute_privacy_sgm.compute_dp_sgd_privacy(n, batch_size, noise_multiplider, epochs, delta)[0])