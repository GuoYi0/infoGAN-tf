# -*- coding: utf-8 -*-
useInfoGan = True
style_size = 32  # 噪声维数
num_category = 10  # 类别数
num_continuous = 0  # 设置2个连续变量
discriminator_lr = 4e-4  # 判别器的初始学习率
generator_lr = 2e-3  # 生成器的初始学习率
categorical_lambda = 1.0
continuous_lambda = 1.0
fix_std = True  # 把连续变量的方差固定为1.0
n_epochs = 40
batch_size = 256
plot_every = 200
ckpt = "ckpt"
weight_decay = 1e-6








