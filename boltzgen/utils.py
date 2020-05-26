import numpy as np
from scipy.spatial.distance import pdist, squareform, cdist
import math

"""
Compute the KSD divergence using samples, adapted from the theano code
"""
# From https://github.com/YingzhenLi/SteinGrad/blob/master/hamiltonian/ksd.py
def KSD(z, Sqx, in_h_square=None):

    # compute the rbf kernel
    K, dimZ = z.shape
    sq_dist = pdist(z)
    pdist_square = squareform(sq_dist)**2
    # use median
    median = np.median(pdist_square)
    h_square = 0.5 * median / np.log(K+1.0)
    if in_h_square is not None:
        h_square = in_h_square
    Kxy = np.exp(- pdist_square / h_square / 2.0)

    # now compute KSD
    Sqxdy = np.dot(Sqx, z.T) - np.tile(np.sum(Sqx * z, 1, keepdims=True), (1, K))
    Sqxdy = -Sqxdy / h_square

    dxSqy = Sqxdy.T
    dxdy = -pdist_square / (h_square ** 2) + dimZ / h_square
    # M is a (K, K) tensor
    M = (np.dot(Sqx, Sqx.T) + Sqxdy + dxSqy + dxdy) * Kxy

    # the following for U-statistic
    M2 = M - np.diag(np.diag(M))
    return np.sum(M2) / (K * (K - 1))

def blockKSD(z, Sqx, num_blocks, h_square):
    K, dimZ = z.shape
    block_step = math.floor(K/num_blocks)
    culm_sum = 0
    for i in range(0, K, block_step):
        for j in range(0, K, block_step):
            zrow = z[i:i+block_step, :]
            zcol = z[j:j+block_step, :]
            Sqxrow = Sqx[i:i+block_step, :]
            Sqxcol = Sqx[j:j+block_step, :]
            pdist_square = cdist(zrow, zcol)**2
            Kxy = np.exp(- pdist_square / h_square / 2.0)
            Sqxdy = np.tile(np.sum(Sqxrow * zrow, 1, keepdims=True),
                (1, block_step)) - np.dot(Sqxrow, zcol.T)
            Sqxdy = Sqxdy / h_square
            dxSqy = (np.dot(Sqxcol, zrow.T) - \
                np.tile(np.sum(Sqxcol * zcol, 1, keepdims=True),
                (1, block_step))).T
            dxSqy = -dxSqy / h_square
            dxdy = -pdist_square / (h_square ** 2) + dimZ / h_square

            M = (np.dot(Sqxrow, Sqxcol.T) + Sqxdy + dxSqy + dxdy) * Kxy

            if i == j:
                M = M - np.diag(np.diag(M))
            culm_sum += np.sum(M)
    return culm_sum / (K*(K-1))
            
def get_median_estimate(z):
    z_block = z[0:1000, :]
    sq_dist = pdist(z_block)
    pdist_square = squareform(sq_dist)**2
    return np.median(pdist_square)

