from PIL import Image
import tensorflow as tf
import numpy as np

im = Image.open("./MNIST_test_image.jpg")
img = np.array(im.resize((28, 28), Image.ANTIALIAS).convert("L"))
data = img.reshape([1, 784])

x = tf.placeholder(tf.float32, [None, 784])
W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))
y = tf.nn.softmax(tf.matmul(x, W) + b)
saver = tf.train.Saver()

with tf.Session() as sess:
    save_path = "./tmp/mnist_softmax.ckpt"
    saver.restore(sess, save_path)
    predictions = sess.run(y, feed_dict={x: data})
    print(predictions[0]);