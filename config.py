# -*- coding: utf-8 -*-
useInfoGan = True
style_size = 62  # 暂且不知道干啥的，可能是噪声维数
num_category = 10  # 类别数
num_continuous = 1  # 设置2个连续变量
discriminator_lr = 2e-4  # 判别器的初始学习率
generator_lr = 1e-3  # 生成器的初始学习率
categorical_lambda = 1.0
continuous_lambda = 1.0
fix_std = True  # 把连续变量的方差固定为1.0
n_epochs = 30
batch_size = 64
plot_every = 200
ckpt = "ckpt"







