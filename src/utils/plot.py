from matplotlib import pyplot as plt

def plotHistory(loss):
    plt.figure()

    plt.plot(loss)
    plt.title('loss through training')
    plt.ylabel('loss')
    plt.xlabel('batch')


def learningCurve(plot_loss, plot_val_loss):
    plt.figure()

    plt.plot(plot_loss, "b")
    plt.plot(plot_val_loss, "r")

def showPlot():
    plt.show()
