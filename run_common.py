
import tensorflow as tf
import numpy as np
import datafactory as df
import model


# prepare data and tf.session
train = df.create_supervised_data('E:\dataset\mnist', reshape=True, one_hot=False)
test_x, test_y = df.load_mnist_test('E:\dataset\mnist', reshape=True, one_hot=False)
sess = tf.Session()

num_epochs = 50000*2
step_epochs = 500
batch_size = 128
learn_rate = 0.001
average_loss = 0
save_path = 'common'

# setup siamese network
siamese = model.Siamese()
siamese.common_loss()
solver = tf.train.AdamOptimizer(learn_rate).minimize(siamese.loss)
saver = tf.train.Saver(tf.trainable_variables())
sess.run(tf.global_variables_initializer())
file = open('{}/train.txt'.format(save_path), 'w')

# summary = tf.summary.merge_all()
# writer = tf.summary.FileWriter(save_path, sess.graph)

# start training
for epochs in range(1,num_epochs+1):
    batch_x1, batch_y1 = train.next_batch(batch_size)
    batch_x2, batch_y2 = train.next_batch(batch_size)
    batch_y = np.reshape((batch_y1 == batch_y2).astype('float'), [batch_size,1])
    _, loss_v = sess.run([solver, siamese.loss], feed_dict={siamese.x1: batch_x1,
                                                            siamese.x2: batch_x2,
                                                            siamese.y: batch_y})
    average_loss += loss_v / step_epochs

    if np.isnan(loss_v):
        print('Model diverged with loss = NaN')
        quit()

    if epochs % step_epochs == 0:
        liner = 'epoch {}/{}, loss {}'.format(epochs, num_epochs, average_loss)
        print(liner), file.writelines(liner+'\n')
        average_loss = 0
        saver.save(sess, '{}/model'.format(save_path), global_step=epochs)

# save embeddings
embed_test_x = sess.run(siamese.o1, {siamese.x1:test_x})
embed_test_y = test_y
np.save('{}/embed_test_x.npy'.format(save_path), embed_test_x)
np.save('{}/embed_test_y.npy'.format(save_path), embed_test_y)

embed_train_x = sess.run(siamese.o1, {siamese.x1:train.images})
embed_train_y = train.labels
np.save('{}/embed_train_x.npy'.format(save_path), embed_train_x)
np.save('{}/embed_train_y.npy'.format(save_path), embed_train_y)

sess.close()
file.close()

