import numpy as  np
import pandas as pd
import matplotlib as plt

def make_data(n_sample=100):
    rnd = np.random.RandomState(42)
    x = rnd.uniform(1,7,size=n_sample)
    y_no_noise = (np.sin(4*x) + x + 5)
    y = (y_no_noise + rnd.normal(size=len(x))/2)
    y *= 5
    return x.reshape(-1,1), y.reshape(-1,1)

x, y = make_data()

print(y.shape)
print(x.shape)