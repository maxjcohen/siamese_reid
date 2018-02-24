from matplotlib import pyplot as plt

def plotHistory(loss):
    plt.figure()

    plt.plot(loss)
    plt.title('loss through training')
    plt.ylabel('loss')
    plt.xlabel('batch')


def showPlot():
    plt.show()
