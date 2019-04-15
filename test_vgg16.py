import tensorflow as tf
import numpy as np
import vgg16
import cv2

img = cv2.imread('./cat.jpg')
batch = img.reshape((1, 224, 224, 3))

with tf.device('/cpu:0'):
    with tf.Session() as sess:
        images = tf.placeholder(tf.float32, shape=(1, 224, 224, 3))
        vgg = vgg16.Vgg16()
        vgg.build(images)
        prob = sess.run(vgg.prob, feed_dict={images:batch})
        print(prob)