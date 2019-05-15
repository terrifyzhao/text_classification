import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), '../'))

from text_cnn.graph import Graph
import tensorflow as tf
from utils.load_data import load_data
from text_cnn import args

x, y = load_data('input/train.csv', data_size=None)
eval_index = int(0.9 * len(x))
x_train, y_train = x[0:eval_index], y[0:eval_index]
x_eval, y_eval = x[eval_index:], y[eval_index:]

x_holder = tf.placeholder(dtype=tf.int32, shape=(None, args.seq_length), name='x')
y_holder = tf.placeholder(dtype=tf.int32, shape=None, name='y')

dataset = tf.data.Dataset.from_tensor_slices((x_holder, y_holder))
dataset = dataset.batch(args.batch_size).repeat(args.epochs)
iterator = dataset.make_initializable_iterator()
next_element = iterator.get_next()

model = Graph()
saver = tf.train.Saver()

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
config.gpu_options.per_process_gpu_memory_fraction = 0.9

with tf.Session(config=config)as sess:
    sess.run(tf.global_variables_initializer())
    sess.run(iterator.initializer, feed_dict={x_holder: x_train, y_holder: y_train})
    steps = int((len(x_train) + args.batch_size - 1) / args.batch_size)
    for epoch in range(args.epochs):
        for step in range(steps):
            x_batch, y_batch = sess.run(next_element)
            _, loss, acc = sess.run([model.train_op, model.loss, model.acc],
                                    feed_dict={model.x: x_batch,
                                               model.y: y_batch,
                                               model.keep_prob: args.keep_prob})
            print('epoch:', epoch, ' step:', step, ' loss: ', loss, ' acc:', acc)

        loss_eval, acc_eval = sess.run([model.loss, model.acc],
                                       feed_dict={model.x: x_eval,
                                                  model.y: y_eval,
                                                  model.keep_prob: 1})
        print('loss_eval: ', loss_eval, ' acc_eval:', acc_eval)
        print('\n')
        saver.save(sess, f'../output/textcnn/textcnn_{epoch}.ckpt')
