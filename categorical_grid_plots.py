import numpy as np
import tensorflow as tf
from PIL import Image


def create_image_strip(images, zoom=1.0, gutter=5):
    num_images, image_height, image_width, channels = images.shape

    if channels == 1:
        images = images.reshape(num_images, image_height, image_width)

    # add a gutter between images
    effective_collage_width = num_images * (image_width + gutter) - gutter

    # use white as background
    start_color = (255, 255, 255)

    collage = Image.new('RGB', (effective_collage_width, image_height), start_color)
    offset = 0
    for image_idx in range(num_images):
        to_paste = Image.fromarray(
            (images[image_idx] * 255).astype(np.uint8)
        )
        collage.paste(
            to_paste,
            box=(offset, 0, offset + image_width, image_height)
        )
        offset += image_width + gutter

    if zoom != 1:
        collage = collage.resize(
            (
                int(collage.size[0] * zoom),
                int(collage.size[1] * zoom)
            ),
            Image.NEAREST
        )
    return np.array(collage)


def create_continuous_noise(num_continuous, style_size, size):
    style = np.random.standard_normal(size=(size, style_size))
    if num_continuous > 0:
        continuous = np.random.uniform(-1.0, 1.0, size=(size, num_continuous))
        return np.hstack([continuous, style])
    return style


class CategoricalPlotter(object):
    def __init__(self,
                 journalist,
                 categorical_cardinality,
                 num_continuous,
                 style_size,
                 generate,
                 row_size=10,
                 zoom=2.0,
                 gutter=3):
        self._journalist = journalist
        self._gutter = gutter
        self.categorical_cardinality = categorical_cardinality
        self.style_size = style_size
        self.num_continuous = num_continuous
        self._generate = generate
        self._zoom = zoom

        self._placeholders = {}
        self._image_summaries = {}

    def generate_categorical_variations(self, session, row_size, iteration=None):
        """
        连续噪声保持不变，只变化类别噪声
        :param session:
        :param row_size:
        :param iteration:
        :return:
        """
        images = []
        continuous_noise = create_continuous_noise(
            num_continuous=self.num_continuous,
            style_size=self.style_size,
            size=row_size
        )
        for i in range(self.categorical_cardinality):
            one_hot = np.zeros((row_size, self.categorical_cardinality), dtype=np.float32)
            one_hot[:, i] = 1.0
            z_c_vectors = np.hstack([one_hot, continuous_noise])
            name = "category_%d" % (i,)
            images.append(
                (create_image_strip(self._generate(session, z_c_vectors), zoom=self._zoom, gutter=self._gutter), name))
        self._add_image_summary(session, images, iteration=iteration)

    def _get_placeholder(self, name):
        if name not in self._placeholders:
            self._placeholders[name] = tf.placeholder(tf.uint8, [None, None, 3])
        return self._placeholders[name]

    def _get_image_summary_op(self, names):
        joint_name = "".join(names)
        if joint_name not in self._image_summaries:
            summaries = []
            for name in names:
                image_placeholder = self._get_placeholder(name)
                decoded_image = tf.expand_dims(image_placeholder, 0)
                image_summary_op = tf.summary.image(
                    name,
                    decoded_image, max_outputs=1
                )
                summaries.append(image_summary_op)
            self._image_summaries[joint_name] = tf.summary.merge(summaries)
        return self._image_summaries[joint_name]

    def _add_image_summary(self, session, images, iteration=None):
        feed_dict = {}
        for image, placeholder_name in images:
            placeholder = self._get_placeholder(placeholder_name)
            feed_dict[placeholder] = image

        summary_op = self._get_image_summary_op(
            [name for _, name in images]
        )
        summary = session.run(
            summary_op, feed_dict=feed_dict
        )

        if iteration is None:
            self._journalist.add_summary(summary)
        else:
            self._journalist.add_summary(summary, iteration)
        self._journalist.flush()

    def generate_continuous_variations(self, session, row_size, variations=3, iteration=None):
        """
        连续变量变化，类别变量不变
        :param session:
        :param row_size:
        :param variations:
        :param iteration:
        :return:
        """
        categorical_noise = np.random.randint(0, self.categorical_cardinality, size=variations)
        continuous_fixed = create_continuous_noise(
            num_continuous=self.num_continuous,
            style_size=self.style_size,
            size=variations
        )
        linear_variation = np.linspace(-1.0, 1.0, row_size)
        images = []

        for contig_idx in range(self.num_continuous):
            for var_idx in range(variations):
                continuous_modified = continuous_fixed[var_idx:var_idx + 1, :].repeat(
                    row_size, axis=0
                )
                # make this continuous variable vary linearly over the row:
                continuous_modified[:, contig_idx] = linear_variation
                one_hot = np.zeros((row_size, self.categorical_cardinality), dtype=np.float32)
                one_hot[:, categorical_noise[var_idx]] = 1.0
                z_c_vectors = np.hstack([one_hot, continuous_modified])
                images.append(
                    (create_image_strip(self._generate(session, z_c_vectors), zoom=self._zoom, gutter=self._gutter),
                     "continuous_variable_%d, variation_%d" % (contig_idx, var_idx)))
        self._add_image_summary(session, images, iteration=iteration)

    def generate_images(self, session, row_size, iteration=None):
        self.generate_categorical_variations(
            session, row_size, iteration=iteration
        )
        if self.num_continuous > 0:
            self.generate_continuous_variations(
                session, row_size, variations=3, iteration=iteration
            )
