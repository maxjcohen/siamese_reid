from matplotlib import pyplot as plt

def plotHistory(loss):
    plt.figure()

    plt.plot(loss)
    plt.title('loss through training')
    plt.ylabel('loss')
    plt.xlabel('batch')


def learningCurve(plot_x, plot_loss, plot_val_loss):
    plt.figure()

    plt.plot(plot_x, plot_loss)
    plt.plot(plot_x, plot_val_loss)

def showPlot():
    plt.show()
