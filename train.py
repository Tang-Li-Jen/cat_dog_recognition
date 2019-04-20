from sklearn.model_selection import StratifiedShuffleSplit
import tensorflow as tf
import numpy as np
import cv2


def generate_label(images_list):
    label = np.array([ 1  if 'cat' in path else 0 for path in train_images_list])
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

num_steps = 1001
batch_size = 32

# with tf.device("/gpu:0"):
#     with tf.Session() as sess:
#     init = tf.initialize_all_variables()
#     sess.run(init)
#     print ("Initialized")
#     for step in range(num_steps):
#         offset = (step * batch_size) % (train_labels.shape[0] - batch_size)
#         batch_data = train_dataset[offset:(offset + batch_size), :, :, :]
#         batch_labels = train_labels[offset:(offset + batch_size), :]
#         feed_dict = {tf_train_dataset : batch_data, tf_train_labels : batch_labels}
#         _, l, predictions = session.run([optimizer, loss, train_prediction], feed_dict=feed_dict)
#         if (step % 50 == 0):
#             print ("Minibatch loss at step", step, ":", l)
#             print ("Minibatch accuracy: %.1f%%" % accuracy(predictions, batch_labels))
#             print ("Validation accuracy: %.1f%%" % accuracy(valid_prediction.eval(), valid_labels))
#             #print ("Test accuracy: %.1f%%" % accuracy(test_prediction.eval(), test_labels))