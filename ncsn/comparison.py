import matplotlib.pyplot as plt

def comparison(data, output):
    fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(10, 4))

    ax1.scatter(data[:,0], data[:,1], s=5, alpha=0.6)
    ax1.set_title("True distribution")
    ax2.scatter(output[:, 0], output[:, 1], s=5, alpha=0.6)
    ax2.set_title("Recreate")
    plt.show()
    # plt.savefig("comparison", dpi=600)