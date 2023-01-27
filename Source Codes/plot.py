import matplotlib.pyplot as plt

def plot_samples(data, channel:int, title=None, plot_name="", n_examples =20):

    rows = int(n_examples / 5)
    plt.figure(figsize=(rows, rows))
    
    if title: plt.suptitle(title)
    X, Y = data
    
    for i in range(n_examples):
        
        ax = plt.subplot(rows, 5, i+1)

        image = 255 - X[i, channel].view((28,28))
        ax.imshow(image)
        ax.axis("off")
        