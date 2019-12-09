import numpy as np


def make_one_hot(indices, size):
    as_one_hot = np.zeros((indices.shape[0], size))
    as_one_hot[np.arange(0, indices.shape[0]), indices] = 1.0
    return as_one_hot


def create_continuous_noise(num_continuous, style_size, size):
    continuous = np.random.uniform(-1.0, 1.0, size=(size, num_continuous))
    style = np.random.standard_normal(size=(size, style_size))
    return np.hstack([continuous, style])


def encode_infogan_noise(categorical_cardinality, categorical_samples, continuous_samples):
    noise = []
    for cardinality, sample in zip(categorical_cardinality, categorical_samples):
        noise.append(make_one_hot(sample, size=cardinality))
    noise.append(continuous_samples)
    return np.hstack(noise)


def create_infogan_noise_sample(num_category, num_continuous, style_size):
    def sample(batch_size):
        noise = list()
        randlabel = np.random.randint(0, num_category, batch_size)
        one_hot = np.zeros((batch_size, num_category), dtype=np.float32)
        one_hot[np.arange(0, batch_size), randlabel] = 1.0
        noise.append(one_hot)
        if num_continuous > 0:
            noise.append(np.random.uniform(-1.0, 1.0, size=(batch_size, num_continuous)))
        noise.append(np.random.standard_normal(size=(batch_size, style_size)))
        return np.hstack(noise)
    return sample


def create_gan_noise_sample(style_size):
    def sample(batch_size):
        return np.random.standard_normal(size=(batch_size, style_size))

    return sample
