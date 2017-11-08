
import gzip as gp
import numpy as np
import os as os
import pickle as pk
import scipy.io as sio

mnist_url = 'http://yann.lecun.com/exdb/mnist/'
train_images = 'train-images-idx3-ubyte.gz'
train_labels = 'train-labels-idx1-ubyte.gz'
test_images = 't10k-images-idx3-ubyte.gz'
test_labels = 't10k-labels-idx1-ubyte.gz'

cifar10_url = 'https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz'
train_batch = ('data_batch_1', 'data_batch_2',
			   'data_batch_3', 'data_batch_4',
			   'data_batch_1')
test_batch = ('test_batch')

svhn_url = 'http://ufldl.stanford.edu/housenumbers/'
full_number_train = 'train.tar.gz'
full_number_test = 'test.tar.gz'
full_number_extra = 'extra.tar.gz'
cropped_digits_train = 'train_32x32.mat'
cropped_digits_test = 'test_32x32.mat'
cropped_digits_extra = 'extra_32x32.mat'


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


def read32(bytestream):
	dt = np.dtype(np.uint32).newbyteorder('>')
	return np.frombuffer(bytestream.read(4), dtype=dt)[0]

def dense_to_one_hot(labels_dense, num_classes):
	# Convert class labels from scalars to one-hot vectors
	num_labels = labels_dense.shape[0]
	index_offset = np.arange(num_labels) * num_classes
	labels_one_hot = np.zeros((num_labels, num_classes), dtype=np.float32)
	labels_one_hot.flat[index_offset + labels_dense.ravel()] = 1
	return labels_one_hot

def shuffle_data(images, labels):
	assert images.shape[0] == labels.shape[0]
	num_examples = images.shape[0]
	perm = np.arange(num_examples)
	np.random.shuffle(perm)
	images = images[perm]
	labels = labels[perm]
	return images, labels

def normalize_mnist_images(data, reshape, norm):
	data = data.astype(np.float32)
	if reshape == True:
		n_samples = data.shape[0]
		data = data.reshape(n_samples, 28*28)
	if norm == True:
		images = np.multiply(data, 1.0/255.0)
	return images

def to_cifar10_images(data):
	n_samples = data.shape[0]
	images = np.reshape(data, [n_samples, 3, 32, 32])
	images = np.swapaxes(images, 1, 2)
	images = np.swapaxes(images, 2, 3)
	return images

def load_mnist_images(file_path):
	with open(file_path, 'rb') as file:
		print('Extracting', file.name)
		with gp.GzipFile(fileobj=file) as bytestream:
			magic = read32(bytestream)
			if magic != 2051:
				raise ValueError('Invalid magic number %d in MNIST image file: %s' %
								 (magic, file.name))
			num_images = read32(bytestream)
			rows = read32(bytestream)
			cols = read32(bytestream)
			buf = bytestream.read(rows * cols * num_images)
			data = np.frombuffer(buf, dtype=np.uint8)
			data = data.reshape(num_images, rows, cols, 1)
			return data

def load_mnist_labels(file_path, one_hot=True, num_classes=10):
	with open(file_path, 'rb') as file:
		print('Extracting', file.name)
		with gp.GzipFile(fileobj=file) as bytestream:
			magic = read32(bytestream)
			if magic != 2049:
				raise ValueError('Invalid magic number %d in MNIST label file: %s' %
								 (magic, file.name))
			num_items = read32(bytestream)
			buf = bytestream.read(num_items)
			labels = np.frombuffer(buf, dtype=np.uint8)
			if one_hot:
				return dense_to_one_hot(labels, num_classes)
			return labels

def load_mnist_train(path, reshape=True, norm_flag=True, one_hot=True):
	images = load_mnist_images(os.path.join(path,train_images))
	images = normalize_mnist_images(images, reshape, norm_flag)
	labels = load_mnist_labels(os.path.join(path,train_labels), one_hot)
	return images, labels

def load_mnist_test(path, reshape=True, norm_flag=True, one_hot=True):
	images = load_mnist_images(os.path.join(path,test_images))
	images = normalize_mnist_images(images, reshape, norm_flag)
	labels = load_mnist_labels(os.path.join(path,test_labels), one_hot)
	return images, labels

def load_cifar10_train(path, reshape=True, norm_flag=True, one_hot=True):
	images = np.empty([50000,3072], dtype=np.float32)
	labels = np.empty([50000,10], dtype=np.uint8)

	for n in range(len(train_batch)):
		with open(path+'/'+train_batch[n], 'rb') as file:
			dict = pk.load(file, encoding='bytes')
			images[n*10000:(n+1)*10000, :] = dict[b'data']
			# Labels: list-->array
			dense_labels = np.array(dict[b'labels'])
			labels[n*10000:(n+1)*10000, :] = dense_to_one_hot(dense_labels, 10)

	if reshape == False:
		images = to_cifar10_images(images)
	if norm_flag == True:
		images = np.multiply(images, 1.0/255.0)
	if one_hot == False:
		labels = np.argmax(labels, axis=1)

	return images, labels

def load_cifar10_test(path, reshape=True, norm_flag=True, one_hot=True):
	images = np.empty([10000,3072], dtype=np.float32)
	labels = np.empty([10000,10], dtype=np.uint8)

	with open(path+'/'+test_batch, 'rb') as file:
		dict = pk.load(file, encoding='bytes')
		images[:,:] = np.multiply(dict[b'data'], 1.0/255.0)
		# Labels: list-->array
		dense_labels = np.array(dict[b'labels'])
		labels[:,:] = dense_to_one_hot(dense_labels, 10)

	if reshape == False:
		images = to_cifar10_images(images)
	if norm_flag == True:
		images = np.multiply(images, 1.0/255.0)
	if one_hot == False:
		labels = np.argmax(labels, axis=1)

	return images, labels

def load_cifar100(path, flag='train', labels='fine', reshape=True, one_hot=True):
	if flag == 'train':
		file_name = 'train.mat'
	else:
		file_name = 'test.mat'

	data = sio.loadmat(os.path.join(path,file_name))
	images = data['data'].astype(np.float32)
	if labels == 'fine':
		labels = data['fine_labels']
		num_labels = 100
	else: # coarse
		labels = data['coarse_labels']
		num_labels = 20

	images = np.multiply(images, 1/255.0)
	n_samples = images.shape[0]
	images = np.reshape(images, [n_samples,-1])

	if reshape == False:
		images = np.reshape(images, [n_samples, 3, 1024])
		images = np.swapaxes(images, 1, 2)
		images = np.reshape(images, [n_samples, 32, 32, 3])
	if one_hot == True:
		labels = dense_to_one_hot(labels, num_labels)

	return images, labels

def load_svhn_cropped_digits(path, flag='train', reshape=True, one_hot=True):
	if flag == 'train':
		file_name = cropped_digits_train
	elif flag == 'test':
		file_name = cropped_digits_test
	else: # 'extra'
		file_name = cropped_digits_extra

	data = sio.loadmat(os.path.join(path,file_name))
	images = data['X'].astype(np.float32)
	labels = data['y']

	images = np.multiply(images, 1/255.0)
	n_samples = images.shape[-1]
	images = np.reshape(images, [-1,n_samples])
	images = np.swapaxes(images, 0, 1) # n, 3072

	if reshape == False:
		images = np.reshape(images, [n_samples, 32, 32, 3])
	if one_hot == True:
		labels = dense_to_one_hot(labels, 10)

	return images, labels

def create_semi_supervised_data(path, data='mnist', num_label=100, reshape=True, one_hot=True):
	if data == 'mnist':
		images, labels = load_mnist_train(path, reshape, one_hot)
	elif data == 'cifar10':
		images, labels = load_cifar10_train(path, reshape, one_hot)
	elif data == 'cifar100':
		images, labels = load_cifar100(path, reshape=reshape, one_hot=one_hot)
	elif data == 'svhn':
		images, labels = load_svhn_cropped_digits(path, reshape=reshape, one_hot=one_hot)
	else:
		raise NotImplementedError

	assert images.shape[0] == labels.shape[0]
	assert images.shape[0] >= num_label

	images, labels = shuffle_data(images, labels)
	x_label = images[:num_label]
	x_unlabel = images[num_label:]
	y_label = labels[:num_label]
	y_unlabel = labels[num_label:]

	return Dataset(x_label, y_label), Dataset(x_unlabel, y_unlabel)

def create_supervised_data(path, data='mnist', validation=False, reshape=True, one_hot=True):
	if data == 'mnist':
		images, labels = load_mnist_train(path, reshape, one_hot=one_hot)
		n_train_count = 50000
	elif data == 'cifar10':
		images, labels = load_cifar10_train(path, reshape, one_hot=one_hot)
		n_train_count = 40000
	elif data == 'cifar100':
		images, labels = load_cifar100(path, reshape=reshape, one_hot=one_hot)
		n_train_count = 40000
	else:
		raise NotImplementedError

	if validation == False:
		return Dataset(images, labels)
	else:
		tr_img = images[:n_train_count]
		tr_lab = labels[:n_train_count]
		vl_img = images[n_train_count:]
		vl_lab = labels[n_train_count:]
		return Dataset(tr_img,tr_lab), Dataset(vl_img,vl_lab)

class Dataset(object):
	def __init__(self, images, labels):
		self.images = images
		self.labels = labels
		self.num_examples = images.shape[0]
		self.epochs_completed = 0
		self.index_in_epoch = 0

	def next_batch(self, batch_size, fake_data=False, shuffle=True):
		start = self.index_in_epoch

		# Shuffle for the first epoch
		if self.epochs_completed == 0 and start == 0 and shuffle:
			perm0 = np.arange(self.num_examples)
			np.random.shuffle(perm0)
			self.images = self.images[perm0]
			self.labels = self.labels[perm0]

		# Go to the next epoch
		if start + batch_size > self.num_examples:
			# Finished epoch
			self.epochs_completed += 1
			# Get the rest examples in this epoch
			rest_num_examples = self.num_examples - start
			images_rest_part = self.images[start:self.num_examples]
			labels_rest_part = self.labels[start:self.num_examples]
			# Shuffle the data
			if shuffle:
				perm = np.arange(self.num_examples)
				np.random.shuffle(perm)
				self.images = self.images[perm]
				self.labels = self.labels[perm]
			# Start next epoch
			start = 0
			self.index_in_epoch = batch_size - rest_num_examples
			end = self.index_in_epoch
			images_new_part = self.images[start:end]
			labels_new_part = self.labels[start:end]
			return np.concatenate((images_rest_part, images_new_part), axis=0), \
				   np.concatenate((labels_rest_part, labels_new_part), axis=0)
		else:
			self.index_in_epoch += batch_size
			end = self.index_in_epoch
			return self.images[start:end], self.labels[start:end]


if __name__ == '__main__':
	# images, labels = load_svhn_cropped_digits('E:\dataset\svhn', reshape=False, one_hot=False)
	# images, labels = load_cifar100('E:\dataset\cifar100', labels='coarse', reshape=False, one_hot=False)
	# import matplotlib.pyplot as plt
	# figure, axs = plt.subplots(10,10)
	# for i in range(10):
	# 	for j in range(10):
	# 		axs[i][j].imshow(images[10*i+j,:,:,:])
	# 		axs[i][j].set_axis_off()
	# 		axs[i][j].set_title('{}'.format(labels[10*i+j]))
	# plt.show()

	import sampler as spl
	import matplotlib.pyplot as plt
	nc = 10
	data = create_supervised_data('E:/dataset/mnist', 'mnist')
	batch_x, batch_y = data.next_batch(50000)
	z = spl.supervised_gaussian_mixture(50000, batch_y, nc, n_dim=2)

	color_list = plt.get_cmap('hsv', nc+1)
	for n in range(nc):
		index = np.where(batch_y[:,n] == 1)[0]
		point = z[index.tolist(),:]
		x = point[:,0]
		y = point[:,1]
		plt.scatter(x, y, color=color_list(n), edgecolors='face')
	plt.show()


	t = 1