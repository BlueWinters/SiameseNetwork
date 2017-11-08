import tensorflow as tf

class Siamese:
    def __init__(self, x_dim=784, y_dim=10):
        self.x_dim = x_dim
        self.x1 = tf.placeholder(tf.float32, [None, x_dim])
        self.x2 = tf.placeholder(tf.float32, [None, x_dim])
        self.tiny = 1e-6

    def common_loss(self):
        def weighted(input, out_dim, name='output'):
            in_dim = input.get_shape()[-1]
            W = tf.get_variable('alpha', dtype=tf.float32, shape=[in_dim, out_dim],
                                initializer=tf.truncated_normal_initializer(stddev=0.1))
            return tf.nn.sigmoid(tf.matmul(input, W))

        self.y = tf.placeholder(tf.float32, [None,1])
        with tf.variable_scope("siamese") as scope:
            self.o1 = self.common_network(self.x1)
            scope.reuse_variables()
            self.o2 = self.common_network(self.x2)
        self.dis = tf.abs(self.o1-self.o2, name='L1-distance')
        self.output = weighted(self.dis, 1, 'output')
        self.loss = tf.reduce_mean(-self.y*tf.log(self.output+self.tiny)
                                   -(1-self.y)*tf.log(1-self.output+self.tiny))
        return self.loss

    def common_network(self, x):
        fc1 = self.full_connect(x, 1024, 'fc1')
        ac1 = tf.nn.relu(fc1)
        fc2 = self.full_connect(ac1, 1024, 'fc2')
        ac2 = tf.nn.relu(fc2)
        fc3 = self.full_connect(ac2, 2, 'fc3')
        return fc3

    def contrastive_loss(self):
        margin = 5.0
        tiny = 1e-6
        self.y = tf.placeholder(tf.float32, [None,1])

        with tf.variable_scope("siamese") as scope:
            self.o1 = self.contrastive_network(self.x1)
            scope.reuse_variables()
            self.o2 = self.contrastive_network(self.x2)

        with tf.name_scope('loss'): # yi*||CNN(p1i)-CNN(p2i)||^2 + (1-yi)*max(0, C-||CNN(p1i)-CNN(p2i)||^2)
            euclid2 = tf.reduce_sum(tf.square(self.o1 - self.o2), 1)
            euclid = tf.sqrt(euclid2 + tiny)
            self.pos = self.y * euclid2
            self.neg = (1-self.y)*tf.square(tf.maximum(margin-euclid, 0))
            self.loss = tf.reduce_sum(self.pos+self.neg)

        return self.loss

    def contrastive_network(self, x):
        fc1 = self.full_connect(x, 1024, "fc1")
        ac1 = tf.nn.relu(fc1)
        fc2 = self.full_connect(ac1, 1024, "fc2")
        ac2 = tf.nn.relu(fc2)
        fc3 = self.full_connect(ac2, 2, "fc3")
        return fc3

    def full_connect(self, input, out_dim, name):
        in_dim = input.get_shape()[-1]
        W = tf.get_variable(name+'W', dtype=tf.float32, shape=[in_dim, out_dim],
                            initializer=tf.truncated_normal_initializer(stddev=0.01))
        b = tf.get_variable(name+'b', dtype=tf.float32, shape=[out_dim],
                            initializer=tf.constant_initializer(0.01))
        fc = tf.matmul(input, W) + b
        return fc

    def restore(self, sess):
        saver = tf.train.Saver(tf.trainable_variables())
        saver.restore(sess, 'triple/model')