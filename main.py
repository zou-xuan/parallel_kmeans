import scipy.io
import numpy as np
from collections import namedtuple
import matplotlib.pyplot as plt


class SeqKmeans:
    def start(self, cluster_size, file_path, threshold):
        training_data = self.read_data(file_path)
        numofdata = training_data.shape[0]
        cluster_index = np.random.random_integers(0, numofdata - 1, cluster_size)
        cluster = training_data[cluster_index]
        membership = np.full(numofdata, -1)
        random_sample = np.random.random_integers(0, numofdata - 1, 10000)
        training_data = training_data[random_sample, :]
        self.seq_kmeans(training_data, cluster, membership, threshold)
        self.plot(cluster)

    def read_data(self, file_path):
        mat = scipy.io.loadmat(file_path)
        print(mat)
        training_data = mat['images'].astype(int)
        # np.sign(training_data)
        num_of_data = training_data.shape[2]
        training_data = training_data.transpose(2, 0, 1).reshape(num_of_data, -1)
        return training_data  # (60000,784)

    def seq_kmeans(self, training_data, cluster, membership, threshold):
        loop = 0
        while True:
            print('loop' + str(loop))
            loop += 1
            cluster_map = {}
            for j in range(cluster.shape[0]):
                cluster_map[j] = []
            delta = self.update_cluster(training_data, cluster_map, cluster, membership)
            cluster = self.find_center(cluster, cluster_map)
            print(delta)
            if delta / training_data.shape[0] < threshold or loop > 500:
                break

    def update_cluster(self, training_data, cluster_map, cluster, membership):
        delta = 0
        for i in range(training_data.shape[0]):
            index = self.find_cluster(training_data[i], cluster)
            if membership[i] != index:
                membership[i] = index
                delta += 1
            try:
                cluster_map[index].append(training_data[i])
            except KeyError:
                cluster_map[index] = [training_data[i]]
        return delta

    def find_cluster(self, current_data, cluster):
        Distance = namedtuple('Distance', 'center dist')
        array = []
        for i in range(cluster.shape[0]):
            array.append(Distance(center=i, dist=np.linalg.norm(current_data - cluster[i])))
        index = sorted(array, key=lambda t: float(t[1]))[0][0]
        return index

    def find_center(self, cluster, cluster_map):
        new_cluster = np.zeros_like(cluster)
        for i in range(cluster.shape[0]):
            new_cluster[i, :] = np.mean(cluster_map[i],axis=0)
        return new_cluster

    def plot(self, cluster):
        fig, axs = plt.subplots(nrows=1, ncols=cluster.shape[0])
        plt_cluster = cluster.reshape(cluster.shape[0], 28, 28)
        for i in range(cluster.shape[0]):
            axs[i].imshow(plt_cluster[i])
            axs[i].axis("off")
        plt.show()


if __name__ == '__main__':
    sq = SeqKmeans()
    sq.start(10, './mnist_data/images.mat', threshold=0.001)
