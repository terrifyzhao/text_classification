import tensorflow as tf
from text_cnn import args


class Graph:
    def __init__(self):
        self.x = tf.placeholder(dtype=tf.int32, shape=(None, args.seq_length), name='p')
        self.y = tf.placeholder(dtype=tf.int32, shape=None, name='y')
        self.keep_prob = tf.placeholder(dtype=tf.float32, name='drop_rate')
        self.embedding = tf.get_variable(dtype=tf.float32, shape=(args.vocab_size, args.char_embedding_size),
                                         name='embedding')
        self.forward()

    def dropout(self, x):
        return tf.nn.dropout(x, keep_prob=self.keep_prob)

    def forward(self):
        x_embedding = tf.nn.embedding_lookup(self.embedding, self.x)

        kernel_height = [2, 3, 4]
        pool_values = []
        for i in range(3):
            conv = tf.layers.conv2d(tf.expand_dims(x_embedding, 3),
                                    filters=2,
                                    kernel_size=(kernel_height[i], args.char_embedding_size),
                                    activation='relu')
            pool_value = tf.layers.max_pooling2d(conv,
                                                 pool_size=(args.seq_length - kernel_height[i] + 1, 1),
                                                 strides=1)
            pool_values.append(pool_value)
        concat_value = tf.concat(pool_values, axis=3)
        value = tf.reshape(concat_value, shape=(-1, concat_value.shape[-1]))
        value = self.dropout(value)
        self.logits = tf.layers.dense(value, 2, activation='relu')
        self.train()

    def train(self):
        y = tf.one_hot(self.y, args.class_size)
        loss = tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=self.logits)
        self.loss = tf.reduce_mean(loss)
        self.train_op = tf.train.AdamOptimizer(args.learning_rate).minimize(self.loss)
        prediction = tf.argmax(self.logits, axis=1)
        correct_prediction = tf.equal(tf.cast(prediction, tf.int32), self.y)
        self.acc = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
