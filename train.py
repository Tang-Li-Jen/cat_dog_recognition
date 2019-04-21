from sklearn.model_selection import StratifiedShuffleSplit
import tensorflow as tf
import numpy as np
import vgg16
import cv2
import os

def generate_label(images_list):
    label = np.array([ 1  if 'cat' in path else 0 for path in train_images_list])
    label = label.reshape((-1, 1))
    return label

def load_image(image_path, mean=vgg_mean):
    image = cv2.imread(image_path)
    
    short_edge = min(image.shape[:2])

    yy = int((image.shape[0] - short_edge) / 2)
    xx = int((image.shape[1] - short_edge) / 2)
    crop_image = image[yy: yy + short_edge, xx: xx + short_edge]
    
    resized_image = cv2.resize(crop_image, (224, 224)) 
    
    return resized_image

def get_batches(x, y, batch_size=32):
    num_rows = y.shape[0]
    num_batches = num_rows // batch_size

    if num_rows % batch_size != 0:
        num_batches = num_batches + 1
    
    for batch in range(num_batches):
        yield x[batch_size * batch: batch_size * (batch + 1)], y[batch_size * batch: batch_size * (batch + 1)]

def accuracy(predictions, labels):
   return (100.0 * np.sum(np.argmax(predictions, 1) == np.argmax(labels, 1)) / predictions.shape[0])

train_path = "./train/"
test_path = "./test/"

train_images_list = [train_path + i for i in os.listdir(train_path)]
labels = generate_label(train_images_list)
images = np.array([load_image(f) for f in train_images_list])

splitter = StratifiedShuffleSplit(n_splits=1, test_size=0.1)
train_indices, val_indices = next(splitter.split(images, labels))

train_images, train_labels = images[train_indices], labels[train_indices]
val_images, val_labels = images[val_indices], labels[val_indices]

num_epochs = 5
batch_size = 32

with tf.device("/gpu:0"):
    with tf.Session() as sess:
        vgg = vgg16.Vgg16("./vgg16.npy")
        vgg.build()

        init = tf.global_variables_initializer()
        sess.run(init)
        print ("Initialized")
        for epoch in range(num_epochs):
            for batch_train_images, batch_train_labels in get_batches(train_images, train_labels, batch_size=batch_size):
                feed_dict = {"images:0" : batch_train_images, "labels:0" : batch_train_labels}
                _, l, predictions = sess.run([vgg.optimizer, vgg.loss, vgg.prediction], feed_dict=feed_dict)
                # if (step % 50 == 0):
                #     print ("Minibatch loss at step", step, ":", l)
                #     print ("Minibatch accuracy: %.1f%%" % accuracy(predictions, batch_labels))
                    # print ("Validation accuracy: %.1f%%" % accuracy(valid_prediction.eval(), valid_labels))
                    #print ("Test accuracy: %.1f%%" % accuracy(test_prediction.eval(), test_labels))