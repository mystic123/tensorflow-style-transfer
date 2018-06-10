import time

import numpy as np
import tensorflow as tf
from PIL import Image
from skimage import color
from skimage.io import imsave

from models.vgg import vgg_16

LEARNING_RATE = 10.0

LOGDIR = './logs/'

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('content', '', 'Content image.')
tf.app.flags.DEFINE_string('style', '', 'Style image.')
tf.app.flags.DEFINE_string('ckpt_file', 'checkpoints/vgg_16.ckpt', 'Checkpoint file.')
tf.app.flags.DEFINE_string('result_file', 'result.jpg', 'Result file.')
tf.app.flags.DEFINE_integer('steps', 1000, 'Number of steps to run.')
tf.app.flags.DEFINE_integer('resize', -1, 'Resize shorter dim of content img to this size.')

IMAGENET_MEAN = [123.68, 116.779, 103.939]

STYLE_LAYERS = ['vgg_16/conv1/conv1_1', 'vgg_16/conv2/conv2_1', 'vgg_16/conv3/conv3_1', 'vgg_16/conv4/conv4_1',
                'vgg_16/conv5/conv5_1']
CONTENT_LAYERS = ['vgg_16/conv4/conv4_2', 'vgg_16/conv5/conv5_2']

CONTENT_WEIGHT = 1
STYLE_WEIGHT = 1e3
TVD_WEIGHT = 1e-2


def resize(img):
    if FLAGS.resize <= 0:
        return img
    width, height = img.size
    ratio = min(width, height) / FLAGS.resize
    if width < height:
        width = FLAGS.resize
        height = int(height / ratio)
    else:
        height = FLAGS.resize
        width = int(width / ratio)
    return img.resize((width, height))


def match_luminance(content, style):
    content = content / 255
    style = style / 255
    content = color.rgb2yiq(content)
    style = color.rgb2yiq(style)
    mean_c = np.mean(content)
    mean_s = np.mean(style)
    stddev_c = np.std(content)
    stddev_s = np.std(style)
    style = (stddev_c / stddev_s) * (style - mean_s) + mean_c
    style = np.clip(color.yiq2rgb(style), 0, 1) * 255
    return style


def main(args):
    content_image = Image.open(FLAGS.content)
    content_image = resize(content_image)
    style_image = Image.open(FLAGS.style).resize(content_image.size)

    content_image = np.asarray(content_image, dtype=np.float32)
    style_image = np.asarray(style_image, dtype=np.float32)

    # match luminance between style and content image
    style_image = match_luminance(content_image, style_image)

    content_image = np.expand_dims(content_image, 0)
    style_image = np.expand_dims(style_image, 0)

    img_shape = content_image.shape
    with tf.name_scope('image'):
        random_image = tf.random_normal(mean=1, stddev=.01, shape=img_shape)
        image = tf.Variable(initial_value=random_image, name='image', dtype=tf.float32)
        tf.summary.image('img', tf.clip_by_value(image, 0, 255), max_outputs=1)
        # subtract mean
        inputs = image - IMAGENET_MEAN
        # convert to BGR, because VGG16 was trained on BGR images
        channels = tf.unstack(inputs, axis=-1)
        inputs = tf.stack([channels[2], channels[1], channels[0]], axis=-1)
    _, endpoints = vgg_16(inputs, is_training=False, scope='vgg_16')

    saver = tf.train.Saver(var_list=tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='vgg_16'))

    style_tensors = [endpoints[l] for l in STYLE_LAYERS]
    content_tensors = [endpoints[l] for l in CONTENT_LAYERS]

    image_style_tensors = [endpoints[l] for l in STYLE_LAYERS]
    image_content_tensors = [endpoints[l] for l in CONTENT_LAYERS]

    with tf.Session() as sess:
        tf.global_variables_initializer().run()
        saver.restore(sess, FLAGS.ckpt_file)
        style_features = sess.run(style_tensors, feed_dict={image: style_image})
        content_features = sess.run(content_tensors, feed_dict={image: content_image})

    # define style loss
    style_losses = []
    for image_layer, style_layer in zip(image_style_tensors, style_features):
        _, height, width, channels = image_layer.get_shape().as_list()
        size = height * width * channels

        # computer gram matrices
        image_feats_reshape = tf.reshape(image_layer, [-1, channels])
        image_gram = tf.matmul(tf.transpose(image_feats_reshape), image_feats_reshape) / size
        style_feats_reshape = tf.reshape(style_layer, [-1, channels])
        style_gram = tf.matmul(tf.transpose(style_feats_reshape), style_feats_reshape) / size

        loss = tf.square(tf.norm(image_gram - style_gram, ord='fro', axis=(0, 1)))
        style_losses.append(loss)

    style_loss = STYLE_WEIGHT * tf.add_n(style_losses)

    # define content loss
    content_losses = []
    for image_layer, content_layer in zip(image_content_tensors, content_features):
        _, height, width, channels = image_layer.get_shape().as_list()
        size = height * width * channels
        loss = tf.nn.l2_loss(image_layer - content_layer) / size
        content_losses.append(loss)

    content_loss = CONTENT_WEIGHT * tf.add_n(content_losses)

    # total variation denoising loss
    tvd_loss = TVD_WEIGHT * tf.reduce_sum(tf.image.total_variation(image))

    loss = style_loss + content_loss + tvd_loss

    global_step = tf.train.get_or_create_global_step()
    optim = tf.train.AdamOptimizer(LEARNING_RATE)
    train_op = optim.minimize(loss, global_step=global_step, var_list=[image])

    with tf.Session() as sess:
        tf.global_variables_initializer().run()
        saver.restore(sess, FLAGS.ckpt_file)

        noise = tf.random_normal(mean=1, stddev=.01, shape=img_shape)
        rand_init = tf.clip_by_value(content_image * noise, 0, 255)
        image.assign(rand_init).eval()
        for step in range(FLAGS.steps):
            t0 = time.time()
            _, style_loss_val, content_loss_val, tvd_loss_val, loss_val = sess.run(
                [train_op, style_loss, content_loss, tvd_loss, loss])
            t = time.time() - t0
            if step % 10 == 0:
                format_str = 'step: {}/{} loss: style: {}, content: {}, tvd: {}, total: {} | time: {:.2f} s/step'
                print(format_str.format(step, FLAGS.steps, style_loss_val, content_loss_val, tvd_loss_val, loss_val, t))

        img = sess.run(image)[0]

    # transfer luminance
    img = np.clip(img, 0, 255) / 255
    content_image = content_image[0] / 255
    result_y = np.expand_dims(color.rgb2yiq(img)[:, :, 0], 2)
    content_iq = color.rgb2yiq(content_image)[:, :, 1:]
    img = np.dstack((result_y, content_iq))
    img = np.clip(color.yiq2rgb(img), 0, 1)
    imsave(FLAGS.result_file, img)


if __name__ == '__main__':
    tf.app.run()
