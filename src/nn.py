import numpy as np


def im2col(img, kh, kw, stride=1):
    H, W, C = img.shape
    ho = (H - kh) // stride + 1
    wo = (W - kw) // stride + 1
    cols = np.zeros((kh * kw * C, ho * wo))
    for y in range(ho):
        for x in range(wo):
            patch = img[y*stride:y*stride+kh, x*stride:x*stride+kw, :]
            cols[:, y * wo + x] = patch.ravel()
    return cols


def conv2d(x, kernel, bias):
    kh, kw, cin, cout = kernel.shape
    ho = x.shape[0] - kh + 1
    wo = x.shape[1] - kw + 1
    cols = im2col(x, kh, kw)
    k = kernel.reshape(-1, cout)
    out = k.T @ cols
    out += bias.reshape(-1, 1)
    return out.reshape(cout, ho, wo).transpose(1, 2, 0)


def maxpool2d(inp, size=2, stride=2):
    H, W, C = inp.shape
    ho = (H - size) // stride + 1
    wo = (W - size) // stride + 1
    out = np.zeros((ho, wo, C))
    for y in range(ho):
        for x in range(wo):
            out[y, x, :] = np.max(
                inp[y*stride:y*stride+size, x*stride:x*stride+size, :], axis=(0, 1)
            )
    return out


def dense(x, kernel, bias):
    return x @ kernel + bias


def relu(x):
    return np.maximum(0, x)


def softmax(x):
    e = np.exp(x - np.max(x))
    return e / e.sum()


def flatten(x):
    return x.ravel()


class EmotionCNN:
    def __init__(self, weights_path):
        import h5py
        f = h5py.File(weights_path, 'r')
        self.w = {}
        for group_name in f:
            grp = f[group_name]
            for inner in grp:
                dset = grp[inner]
                for dset_name in dset:
                    full_key = f'{group_name}_{dset_name}'
                    self.w[full_key] = dset[dset_name][()]
        f.close()

    def predict(self, img):
        if img.ndim == 2:
            img = img[..., np.newaxis]
        x = img.astype(np.float32) / 255.0

        x = conv2d(x, self.w['conv2d_1_kernel:0'], self.w['conv2d_1_bias:0'])
        x = relu(x)
        x = conv2d(x, self.w['conv2d_2_kernel:0'], self.w['conv2d_2_bias:0'])
        x = relu(x)
        x = maxpool2d(x)
        x = conv2d(x, self.w['conv2d_3_kernel:0'], self.w['conv2d_3_bias:0'])
        x = relu(x)
        x = maxpool2d(x)
        x = conv2d(x, self.w['conv2d_4_kernel:0'], self.w['conv2d_4_bias:0'])
        x = relu(x)
        x = maxpool2d(x)

        x = flatten(x)
        x = dense(x, self.w['dense_1_kernel:0'], self.w['dense_1_bias:0'])
        x = relu(x)
        x = dense(x, self.w['dense_2_kernel:0'], self.w['dense_2_bias:0'])
        return softmax(x)
