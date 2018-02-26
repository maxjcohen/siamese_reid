from matplotlib import pyplot as plt

def plotHistory(loss, title=""):
    plt.figure()

    plt.plot(loss)
    plt.title(title)
    plt.ylabel('loss')
    plt.xlabel('batch')

def plotCMC(ranks):
    plt.figure()

    plt.plot(ranks)

    for index, value in enumerate(ranks):
        plt.annotate("rg{}: {:.2f}".format(index+1, value), (index, value), xytext=(index, value))

    plt.title("CMC")
    plt.ylabel('fraction')
    plt.xlabel('rank')

def learningCurve(plot_loss, plot_val_loss):
    plt.figure()

    plt.plot(plot_loss, "b")
    plt.plot(plot_val_loss, "r")

def showPlot():
    plt.show()
