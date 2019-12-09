# -*- coding: utf-8 -*-
import numpy as np
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import config as cfg
from noise_utils import create_infogan_noise_sample, create_gan_noise_sample
import tensorflow.layers as layers
from tf_utils import scope_variables
from os.path import join, realpath, dirname, basename, exists
from categorical_grid_plots import CategoricalPlotter
import progressbar
from random import shuffle

SCRIPT_DIR = dirname(realpath(__file__))
PROJECT_DIR = SCRIPT_DIR


def load_mnist_dataset():
    mnist = input_data.read_data_sets("mnist", one_hot=True)
    pixel_height = 28
    pixel_width = 28
    n_channels = 1
    for dset in [mnist.train, mnist.validation, mnist.test]:
        num_images = len(dset.images)
        dset.images.shape = (num_images, pixel_height, pixel_width, n_channels)
    return mnist.train.images, mnist.train.labels


def generatorNet(name, inputs, is_training, use_batchNorm, reuse=None):
    idx = 0
    f = inputs
    with tf.variable_scope(name, reuse=reuse):
        f = layers.dense(f, 1024, None, name="dense_%d" % idx)
        if use_batchNorm:
            f = layers.batch_normalization(f, training=is_training, name="bn_%d" % idx)
        f = tf.nn.relu(f, "relu_%d" % idx)

        idx += 1
        f = layers.dense(f, 7 * 7 * 128, None, name="dense_%d" % idx)  # 6272
        if use_batchNorm:
            f = layers.batch_normalization(f, training=is_training, name="bn_%d" % idx)
        f = tf.nn.relu(f, "relu_%d" % idx)

        f = tf.reshape(f, [-1, 7, 7, 128], name="reshape_%d" % idx)

        idx += 1
        f = layers.conv2d_transpose(f, 64, kernel_size=4, strides=2, padding="SAME", name="deconv_%d" % idx)
        if use_batchNorm:
            f = layers.batch_normalization(f, training=is_training, name="bn_%d" % idx)
        f = tf.nn.relu(f, "relu_%d" % idx)

        idx += 1
        f = layers.conv2d_transpose(f, 1, kernel_size=4, strides=2, padding="SAME", name="deconv_%d" % idx)
        f = tf.nn.sigmoid(f, "sigmoid_%d" % idx)

    return f


def discriminator_forward(
        name, inputs, is_training, use_batchNorm=True, reuse=None):
    with tf.variable_scope(name, reuse=reuse):
        out = discriminatorNet(inputs, is_training, use_batchNorm)
        prob = layers.dense(out, 1, activation=tf.nn.sigmoid, name="discriminator_out")
    return {"prob": prob, "hidden": out}


def discriminatorNet(inputs, is_training, use_batchNorm):
    idx = 0
    f = inputs
    f = layers.conv2d(f, 64, kernel_size=4, strides=2, padding="SAME", name="conv_%d" % idx)
    if use_batchNorm:
        f = layers.batch_normalization(f, training=is_training, name="bn_%d" % idx)
    f = tf.nn.leaky_relu(f, alpha=0.01, name="lrelu_%d" % idx)

    idx += 1
    f = layers.conv2d(f, 128, kernel_size=4, strides=2, padding="SAME", name="conv_%d" % idx)
    if use_batchNorm:
        f = layers.batch_normalization(f, training=is_training, name="bn_%d" % idx)
    f = tf.nn.leaky_relu(f, alpha=0.01, name="lrelu_%d" % idx)

    idx += 1
    f = layers.flatten(f)
    f = layers.dense(f, 1024, name="dense_%d" % idx)
    f = tf.nn.leaky_relu(f, alpha=0.01, name="lrelu_%d" % idx)
    return f


def reconstruct_mutual_info(true_categoricals,
                            true_continuous,
                            true_labels,
                            categorical_lambda,
                            continuous_lambda,
                            fix_std,
                            hidden,
                            is_training,
                            reuse=None,
                            name="mutual_info"):
    """
    衡量输入与Q分支输出之间的互信息。返回一个需要越大越好的值
    :param true_categoricals: 输入的真实类别标签， placeholder
    :param true_continuous:  输入的连续变量
    :param categorical_lambda:
    :param continuous_lambda:
    :param fix_std:  是否把连续变量固定为1.0
    :param hidden: 判别器的倒数第二层
    :param is_training: 是否训练
    :param reuse:
    :param name:
    :return:
    """
    f = hidden
    with tf.variable_scope(name, reuse=reuse):
        f = layers.dense(hidden, 128)
        # f = layers.batch_normalization(f, name='q_bn', training=is_training)
        f = tf.nn.leaky_relu(f, 0.01)
        num_categorical = sum([true_categorical.get_shape()[1].value for true_categorical in true_categoricals])  # 10
        num_continuous = true_continuous.get_shape()[1].value  # 2
        f = layers.dense(f, units=num_categorical + (num_continuous if fix_std else (num_continuous * 2)))

        # distribution logic
        offset = 0
        ll_categorical = tf.constant(0.0, dtype=tf.float32)
        for true_categorical in true_categoricals:
            cardinality = true_categorical.get_shape()[1].value
            prob_categorical = tf.nn.softmax(f[:, offset:offset + cardinality])
            ll_categorical_new = tf.reduce_sum(tf.log(prob_categorical + 1e-6) * true_categorical, axis=1, )  # 负交叉熵
            ll_categorical += ll_categorical_new
            offset += cardinality
        mean_contig = f[:, num_categorical:num_categorical + num_continuous]

        if fix_std:
            std_contig = tf.ones_like(mean_contig)
        else:
            std_contig = tf.sqrt(tf.exp(f[:, num_categorical + num_continuous:num_categorical + num_continuous * 2]))
        epsilon = (true_continuous - mean_contig) / (std_contig + 1e-6)
        ll_continuous = tf.reduce_sum(-0.5 * np.log(2 * np.pi) - tf.log(std_contig + 1e-6) - 0.5 * tf.square(epsilon),
                                      axis=1)  # 正态分布
        mutual_info_lb = continuous_lambda * ll_continuous + categorical_lambda * ll_categorical
    return {
        "mutual_info": tf.reduce_mean(mutual_info_lb),
        "ll_categorical": tf.reduce_mean(ll_categorical),
        "ll_continuous": tf.reduce_mean(ll_continuous),
        "std_contig": tf.reduce_mean(std_contig)
    }


def main():
    X = load_mnist_dataset()
    dataset_name = "mnist"
    z_size = cfg.style_size + cfg.num_category + cfg.num_continuous
    sample_noise = create_infogan_noise_sample(cfg.num_category, cfg.num_continuous, cfg.style_size)
    # sample_noise(8),返回一个shape为(8, 74)的噪声数据，
    # 前十个为 one hot编码的均匀分布的0~9类别采用，然后是两个均匀分布的数，然后是62个正态分布的数
    discriminator_lr = tf.get_variable(
        "discriminator_lr", (),
        initializer=tf.constant_initializer(cfg.discriminator_lr)
    )
    generator_lr = tf.get_variable(
        "generator_lr", (),
        initializer=tf.constant_initializer(cfg.generator_lr)
    )
    n_images, image_height, image_width, n_channels = X[0].shape
    print("total images: ", n_images)
    true_images = tf.placeholder(tf.float32, [None, image_height, image_width, n_channels], name="true_images")
    true_labels = tf.placeholder(tf.float32, shape=[None, cfg.num_category], name="true_labels")
    zc_vectors = tf.placeholder(tf.float32, [None, z_size], name="zc_vectors")  # 输入进生成器的噪声
    is_training_discriminator = tf.placeholder(tf.bool, [], name="is_training_discriminator")
    is_training_generator = tf.placeholder(tf.bool, [], name="is_training_generator")

    fake_images = generatorNet(
        name="generator", inputs=zc_vectors, use_batchNorm=True, is_training=is_training_generator)






    discriminator_fake = discriminator_forward(
        name="discriminator", inputs=fake_images, is_training=is_training_discriminator, use_batchNorm=True, reuse=None)
    prob_fake = discriminator_fake["prob"]
    discriminator_true = discriminator_forward(
        name="discriminator", inputs=true_images, is_training=is_training_discriminator, use_batchNorm=True, reuse=True)
    prob_true = discriminator_true["prob"]

    # 训练判别器，需要最小化的一些量
    ll_believing_fake_images_are_fake = -tf.log(1.0 - prob_fake + 1e-6)
    ll_true_images = -tf.log(prob_true + 1e-6)

    categorical_c_vectors = []
    offset = 0
    for cardinality in cfg.categorical_cardinality:
        categorical_c_vectors.append(zc_vectors[:, offset:offset + cardinality])
        offset += cardinality
    continuous_c_vector = zc_vectors[:, offset:offset + cfg.num_continuous]
    q_output = reconstruct_mutual_info(
        categorical_c_vectors,
        continuous_c_vector,
        true_labels,
        categorical_lambda=cfg.categorical_lambda,
        continuous_lambda=cfg.continuous_lambda,
        fix_std=cfg.fix_std,
        hidden=discriminator_fake["hidden"],
        is_training=is_training_discriminator,
        name="mutual_info"
    )
    discriminator_obj = tf.reduce_mean(ll_believing_fake_images_are_fake) + tf.reduce_mean(ll_true_images)




    # generator should maximize:
    ll_believing_fake_images_are_real = tf.reduce_mean(tf.log(prob_fake + 1e-6))
    generator_obj = ll_believing_fake_images_are_real

    # discriminator_solver = tf.train.AdamOptimizer(learning_rate=discriminator_lr, beta1=0.5)
    # generator_solver = tf.train.AdamOptimizer(learning_rate=generator_lr, beta1=0.5)
    discriminator_solver = tf.train.AdamOptimizer(learning_rate=discriminator_lr, beta1=0.5)
    generator_solver = tf.train.AdamOptimizer(learning_rate=generator_lr, beta1=0.5)

    discriminator_variables = scope_variables("discriminator")
    generator_variables = scope_variables("generator")

    train_discriminator = discriminator_solver.minimize(-discriminator_obj, var_list=discriminator_variables)
    train_generator = generator_solver.minimize(-generator_obj, var_list=generator_variables)
    discriminator_obj_summary = tf.summary.scalar("discriminator_objective", discriminator_obj)
    generator_obj_summary = tf.summary.scalar("generator_objective", generator_obj)

    if cfg.useInfoGan:

        mutual_info_objective = q_output["mutual_info"]
        mutual_info_variables = scope_variables("mutual_info")
        neg_mutual_info_objective = -mutual_info_objective
        train_mutual_info = generator_solver.minimize(
            neg_mutual_info_objective,
            var_list=generator_variables + discriminator_variables + mutual_info_variables
        )
        ll_categorical = q_output["ll_categorical"]
        ll_continuous = q_output["ll_continuous"]
        std_contig = q_output["std_contig"]

        mutual_info_obj_summary = tf.summary.scalar("mutual_info_objective", mutual_info_objective)
        ll_categorical_obj_summary = tf.summary.scalar("ll_categorical_objective", ll_categorical)
        ll_continuous_obj_summary = tf.summary.scalar("ll_continuous_objective", ll_continuous)
        std_contig_summary = tf.summary.scalar("std_contig", std_contig)
        generator_obj_summary = tf.summary.merge([
            generator_obj_summary,
            mutual_info_obj_summary,
            ll_categorical_obj_summary,
            ll_continuous_obj_summary,
            std_contig_summary
        ])
    else:
        neg_mutual_info_objective = tf.no_op()
        train_mutual_info = tf.no_op()

    log_dir = next_unused_name(
        join(
            PROJECT_DIR,
            "%s_log" % (dataset_name,),
            "infogan" if cfg.useInfoGan else "gan"
        )
    )
    journalist = tf.summary.FileWriter(log_dir, flush_secs=10)
    img_summaries = {}
    if cfg.useInfoGan:
        plotter = CategoricalPlotter(
            journalist=journalist,
            categorical_cardinality=cfg.categorical_cardinality,
            num_continuous=cfg.num_continuous,
            style_size=cfg.style_size,
            generate=lambda s, x: s.run(fake_images, {zc_vectors: x, is_training_discriminator: False,
                                                      is_training_generator: False}))
    else:
        plotter = None
        img_summaries["fake_images"] = tf.summary.image("fake images", fake_images, max_images=10)
    image_summary_op = tf.summary.merge(list(img_summaries.values())) if len(img_summaries) else tf.no_op()
    idxes = np.arange(n_images, dtype=np.int32)
    iters = 0
    saver = tf.train.Saver(max_to_keep=100)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for epoch in range(cfg.n_epochs):
            disc_epoch_obj = []
            gen_epoch_obj = []
            infogan_epoch_obj = []

            shuffle(idxes)
            pbar = create_progress_bar("epoch %d >> " % (epoch,))
            for idx in pbar(range(0, n_images, cfg.batch_size)):
                batch = X[0][idxes[idx: idx + cfg.batch_size]]  # true image
                labels = X[1][idxes[idx: idx + cfg.batch_size]]  # true image
                noise = sample_noise(cfg.batch_size)
                # 训练判别器和互信息
                _, _, summary_result1, disc_obj, infogan_obj = sess.run(
                    [train_discriminator, train_mutual_info, discriminator_obj_summary, discriminator_obj,
                     neg_mutual_info_objective],
                    feed_dict={
                        true_images: batch,
                        zc_vectors: noise,
                        is_training_discriminator: True,
                        is_training_generator: True
                    }
                )
                disc_epoch_obj.append(disc_obj)
                if cfg.useInfoGan:
                    infogan_epoch_obj.append(infogan_obj)

                # 训练生成器和互信息
                noise = sample_noise(cfg.batch_size)
                _, _, summary_result2, gen_obj, infogan_obj = sess.run(
                    [train_generator, train_mutual_info, generator_obj_summary, generator_obj,
                     neg_mutual_info_objective],
                    feed_dict={
                        zc_vectors: noise,
                        is_training_discriminator: True,
                        is_training_generator: True
                    }
                )

                journalist.add_summary(summary_result1, iters)
                journalist.add_summary(summary_result2, iters)
                journalist.flush()
                gen_epoch_obj.append(gen_obj)
                if cfg.useInfoGan:
                    infogan_epoch_obj.append(infogan_obj)
                iters += 1

                if iters % cfg.plot_every == 0:
                    if cfg.useInfoGan:
                        plotter.generate_images(sess, 10, iteration=iters)
                    else:
                        noise = sample_noise(cfg.batch_size)
                        current_summary = sess.run(
                            image_summary_op,
                            {
                                zc_vectors: noise,
                                is_training_discriminator: False,
                                is_training_generator: False
                            }
                        )
                        journalist.add_summary(current_summary, iters)
                    journalist.flush()
                    ckpt_file = join(cfg.ckpt, "gan")
                    saver.save(sess, cfg.ckpt, ckpt_file, iters)

            msg = "epoch %d >> discriminator LL %.2f (lr=%.6f), generator LL %.2f (lr=%.6f)" % (
                epoch,
                np.mean(disc_epoch_obj), sess.run(discriminator_lr),
                np.mean(gen_epoch_obj), sess.run(generator_lr)
            )
            if cfg.useInfoGan:
                msg = msg + ", infogan loss %.2f" % (np.mean(infogan_epoch_obj),)
            print(msg)


def create_progress_bar(message):
    widgets = [
        message,
        progressbar.Counter(),
        ' ',
        progressbar.Percentage(),
        ' ',
        progressbar.Bar(),
        progressbar.AdaptiveETA()
    ]
    pbar = progressbar.ProgressBar(widgets=widgets)
    return pbar


def next_unused_name(name):
    save_name = name
    name_iteration = 0
    while exists(save_name):
        save_name = name + "-" + str(name_iteration)
        name_iteration += 1
    return save_name


if __name__ == "__main__":
    tf.logging.set_verbosity(tf.logging.INFO)
    main()
