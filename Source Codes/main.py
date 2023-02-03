from load_mnist import *
from plot import *

mnist_non_iid_train_dls, mnist_non_iid_test_dls = get_MNIST("non_iid",
    n_samples_train =200, n_samples_test=100, n_clients =5, 
    batch_size =25, shuffle =True)

plot_samples(next(iter(mnist_non_iid_train_dls[0])), 0, "Client 1")
plot_samples(next(iter(mnist_non_iid_train_dls[1])), 0, "Client 2")
plot_samples(next(iter(mnist_non_iid_train_dls[2])), 0, "Client 3")
plot_samples(next(iter(mnist_non_iid_train_dls[1])), 0, "Client 4")
plot_samples(next(iter(mnist_non_iid_train_dls[2])), 0, "Client 5")