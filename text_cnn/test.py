import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), '../'))

from text_cnn.graph import Graph
import tensorflow as tf
from utils.load_data import load_data

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

x, y = load_data('input/test.csv', data_size=None)

model = Graph()
saver = tf.train.Saver()

with tf.Session()as sess:
    sess.run(tf.global_variables_initializer())
    saver.restore(sess, '../output/textcnn/textcnn_49.ckpt')
    loss, acc = sess.run([model.loss, model.acc],
                         feed_dict={model.x: x,
                                    model.y: y,
                                    model.keep_prob: 1})

    print('loss: ', loss, ' acc:', acc)
