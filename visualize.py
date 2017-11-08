import tensorflow as tf
import model
import datafactory as df
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import offsetbox


def visualize_with_images(loss_type='triple'):
    # two ways of visualization: scale to fit [0,1] scale
    # feat = embed - np.min(embed, 0)
    # feat /= np.max(feat, 0)
    # two ways of visualization: leave with original scale
    test_x, test_y = df.load_mnist_test('E:/dataset/mnist', reshape=False)
    x_in = test_x.reshape([-1, 28, 28])
    embed = np.fromfile('{}/embed'.format(loss_type), dtype=np.float32)
    embed = embed.reshape([-1, 2])

    feat = embed
    ax_min = np.min(embed,0)
    ax_max = np.max(embed,0)
    ax_dist_sq = np.sum((ax_max-ax_min)**2)

    plt.figure()
    ax = plt.subplot(111)
    shown_images = np.array([[1., 1.]])
    for i in range(feat.shape[0]):
        dist = np.sum((feat[i] - shown_images)**2, 1)
        if np.min(dist) < 3e-4*ax_dist_sq:   # don't show points that are too close
            continue
        shown_images = np.r_[shown_images, [feat[i]]]
        image_box = offsetbox.AnnotationBbox(
            offsetbox.OffsetImage(x_in[i], zoom=0.5, cmap=plt.cm.gray_r),
            xy=feat[i], frameon=False)
        ax.add_artist(image_box)

    plt.axis([ax_min[0], ax_max[0], ax_min[1], ax_max[1]])
    plt.xticks([]), plt.yticks([])
    plt.title('Embedding from the last layer of the network')
    plt.show()
    plt.savefig('{}/siamese_mnist_test.png'.format(loss_type))

def visual_with_dot(loss_type='contrastive', data_type='test'): # or common
    all_points = np.load('{}/embed_{}_x.npy'.format(loss_type, data_type))
    all_labels = np.load('{}/embed_{}_y.npy'.format(loss_type, data_type))

    color_list = plt.get_cmap('hsv', 10+1)
    for n in range(10):
        index = np.where(all_labels[:] == n)[0]
        points = all_points[index.tolist(),:]
        x = points[:,0]
        y = points[:,1]
        plt.scatter(x, y, color=color_list(n), edgecolors='face')
    plt.xticks([]), plt.yticks([])
    plt.title('{} loss: embedding of {} data'.format(loss_type, data_type))
    plt.savefig('{}/siamese_mnist_{}.jpg'.format(loss_type, data_type))
    plt.show()




if __name__ == "__main__":
    visual_with_dot('common', 'train')