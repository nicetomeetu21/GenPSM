import numpy as np

def gen_masks(surfaces, h=640):
    # print(surfaces.shape)
    surfaces = surfaces.astype(int)
    color_map =[0, 43, 85, 128, 170, 213, 255]
    n,w = surfaces.shape
    ret = np.zeros([h, w], np.uint8)
    for j in range(w):
        for k in range(n):
            if k < n-1:
                ret[surfaces[k][j]:surfaces[k+1][j],j]=color_map[k+1]
            else:
                surfaces[k][j] = np.max(surfaces[:,j])
                ret[surfaces[k][j]:,j]=color_map[k+1]
    return ret

def layer_diff2img(diff_mat):
    # print(diff_mat.shape)
    ret = np.zeros((8, 400))
    ret[7] = 640
    for i in range(6, -1, -1):
        diff_mat[i] = np.where(diff_mat[i] < 0, 0, diff_mat[i])
        diff_mat[i] = np.where(diff_mat[i] > ret[i + 1], ret[i + 1], diff_mat[i])
        ret[i] = ret[i + 1] - diff_mat[i]

    mask = gen_masks(ret[1:-1, :])
    return mask

import numpy as np
def meanfilt(x, k):
    """Apply a length-k mean filter to a 1D array x.
    Boundaries are extended by repeating endpoints.
    """



    assert k % 2 == 1, "Median filter length must be odd."
    assert x.ndim == 1, "Input must be one-dimensional."

    k2 = (k - 1) // 2
    y = np.zeros((len(x), k), dtype=x.dtype)
    y[:, k2] = x
    for i in range(k2):
        j = k2 - i
        y[j:, i] = x[:-j]
        y[:j, i] = x[0]
        y[:-j, -(i + 1)] = x[j:]
        y[-j:, -(i + 1)] = x[-1]
    return np.mean(y, axis=1)

def layer_code2img(code_mat):
    mask = gen_masks(code_mat[1:-1, :])
    return mask

def layer_code2img2(code_mat):
    code_mat = code_mat[1:-1, :]
    n,w = code_mat.shape
    for i in range(n):
        code_mat[i,:] = meanfilt(code_mat[i,:], 3)
    mask = gen_masks(code_mat)
    return mask