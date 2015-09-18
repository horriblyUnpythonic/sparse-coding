import random
import numpy as np
import scipy.io
import matplotlib.pyplot as plt

import logging
import parsedata
import sparsecoding

logging.basicConfig(level=logging.WARNING)


callback_count = [0]

def plt_callback(X, B, S):
    plt.subplot(2, 2, 1)
    plt.plot(X)
    plt.title("originals")
    plt.subplot(2, 2, 2)
    plt.plot(B)
    plt.title("bases")
    plt.subplot(2, 2, 3)
    plt.plot(np.dot(B, S))
    plt.title("reconstructions")
    plt.subplot(2, 2, 4)
    plt.plot(X - np.dot(B, S))
    plt.title("differences")
    callback_count[0] += 1
    plt.savefig('./plots/plot{}.png'.format(callback_count[0]))
    # plt.draw()

patch_size = 16
num_patches = 4
columns = []


use_cache = True
callback = plt_callback

for i in xrange(num_patches):
    if use_cache:
        data = parsedata.get_random_cached_chunks()
        columns.append(data.__array__().reshape((patch_size**2, 1)))
    else:
        while True:
            data = parsedata.get_random_vector_norm_chunks(patch_size**2)
            s = data.std()
            print s
            if s > .01:
                h = random.randint(0,10000)
                fn = '/home/scip/Desktop/stdchunks/std.01-{}.npy'.format(h)
                np.save(fn, data)
            if s > .001:
                columns.append(data.__array__().reshape((patch_size**2, 1)))
                break
X = np.hstack(columns)

plt.plot(X)
plt.show()

# test callback function on svd
#svd = np.linalg.svd(X, full_matrices=False)
#print [x.shape for x in svd]
#callback(X, svd[0], np.dot(np.diag(svd[1]), svd[2]))

num_bases = 64
sparsecoding.sparse_coding(X, num_bases, 0.4, 100, lambda B, S: callback(X, B, S))
