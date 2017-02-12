import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from pandas.tools.plotting import parallel_coordinates
from pandas.tools.plotting import andrews_curves
from pandas.tools.plotting import radviz
from sklearn import preprocessing


def data_plot():
    matplotlib.style.use('ggplot')
    #
    data = [[11.2676056338, 21.8309859155, 33.8028169014, 21.1267605634, 11.2676056338],
            [30.2816901408, 33.0985915493, 15.4929577465, 15.4929577465, 4.92957746479],
            [24.6478873239, 26.7605633803, 20.4225352113, 24.6478873239, 2.81690140845],
            [45.7746478873, 21.8309859155, 10.5633802817, 18.3098591549, 2.81690140845],
            [39.7163120567, 28.3687943262, 18.4397163121, 11.3475177305, 2.12765957447]]

    df2 = pd.DataFrame(np.asarray(data),
                       columns=['Level 0', 'Level 1',
                                'Level 3', 'Level 6', 'Level 9', ], index=['2G', '3G', '4G', 'Wifi', 'Clustering Avg.'])

    plt.figure()
    plt.xlabel('Network Condition')
    plt.ylabel('Configuration Percentage')
    df2.plot.bar()
    plt.show()


def data_plot2():
    matplotlib.style.use('ggplot')
    #
    data = [[28.3687943262, 39.7163120567, 11.3475177305, 18.4397163121, 2.12765957447],
            [21.8309859155, 45.7746478873, 10.5633802817, 18.3098591549, 2.81690140845]]

    df2 = pd.DataFrame(np.asarray(data),
                       columns=['Level 0', 'Level 1',
                                'Level 3', 'Level 6', 'Level 9', ], index=['Clustering Percentage', 'Real Percentage'])

    plt.figure()
    plt.xlabel('Network Condition')
    plt.ylabel('Configuration Percentage')
    df2.plot.bar()
    plt.show()


def data_plot3():
    matplotlib.style.use('ggplot')
    #
    data = [[0.90322581, 0.91129032, 0.94354839, 0.87903226, 0.91129032, 0.89516129,
             0.87096774, 0.91935484, 0.84677419, 0.89516129],
            [0.87903226, 0.92741935, 0.93548387, 0.87903226, 0.91129032, 0.91129032,
             0.87096774, 0.91129032, 0.87903226, 0.84677419],
            [0.5483871, 0.59677419, 0.56451613, 0.66129032, 0.58870968, 0.61290323,
             0.62903226, 0.62903226, 0.58064516, 0.58064516],
            [0.84677419, 0.91935484, 0.94354839, 0.88709677, 0.91935484, 0.90322581,
             0.87096774, 0.93548387, 0.88709677, 0.83870968],
            [0.88709677, 0.93548387, 0.91129032, 0.87096774, 0.91935484, 0.86290323,
             0.91935484, 0.91935484, 0.87096774, 0.86290323]]

    plt.xlabel("0: SVM, 1: Decision Trees, \n2: Neural Networks, 3: Random Forests, 4: k-NN")
    plt.ylabel("Score")
    plt.title("Cross Validation Scores")

    dim = len(data[0])
    w = 0.75
    dimw = w / dim
    x_labels = ["SVM", "D-Trees", "Neural Networks", "Random Forests", "K-NN"]
    x = np.arange(len(data))
    for i in range(len(data[0])):
        y = [d[i] for d in data]
        b = plt.bar(x + i * dimw, y, dimw, bottom=0.001)

    plt.xticks(x + dimw / 2, map(str, x))
    plt.show()

def data_plot4():
    matplotlib.style.use('ggplot')
    #
    data = [[0.96062992126],
            [0.944881889764],
            [0.765590551181],
            [0.952755905512],
            [0.956692913386]]

    # plt.xlabel("0: SVM, 1: Decision Trees, \n2: Neural Networks, 3: Random Forests, 4: k-NN")
    plt.ylabel("Score")
    plt.title("Accuracy of Algorithms (Average Values)")

    dim = len(data[0])
    w = 0.75
    dimw = w / dim / 2
    x_labels = ["SVM", "D-Trees", "NN", "RF", "K-NN"]
    x = np.arange(len(data))
    colors = ['#007acc', '#33cc00', '#e6e600', 'c', '#ff4d4d']
    for i in xrange(len(data[0])):
        y = [d[i] for d in data]
        b = plt.bar(x + i * dimw, y, dimw, bottom=0.001, color=colors)

    plt.xticks(x, x_labels)
    plt.show()


def data_plot5():
    matplotlib.style.use('ggplot')
    #
    data = [[0.96062992126],
            [0.944881889764],
            [0.765590551181],
            [0.952755905512],
            [0.956692913386]]

    # plt.xlabel("0: SVM, 1: Decision Trees, \n2: Neural Networks, 3: Random Forests, 4: k-NN")
    plt.ylabel("Score")
    plt.title("Accuracy of Algorithms (Average Values)")

    dim = len(data[0])
    w = 0.75
    dimw = w / dim / 2
    x_labels = ["SVM", "D-Trees", "NN", "RF", "K-NN"]
    x = np.arange(len(data))
    colors = ['#007acc', '#33cc00', '#e6e600', 'c', '#ff4d4d']
    for i in xrange(len(data[0])):
        y = [d[i] for d in data]
        b = plt.bar(x + i * dimw, y, dimw, bottom=0.001, color=colors)

    plt.xticks(x, x_labels)
    plt.show()

def data_plot6():
    matplotlib.style.use('ggplot')
    #
    data = [[0.785076140828, 0.776214304112, 0.777804968624, 0.8536860897, 0.753165881225, 0.752392879451, 0.752555066052, 0.75288598545],
            [0.885055332928, 0.908986845126, 0.888551455517, 1.35020053263, 0.784572814379, 0.784286219478, 0.78246609046, 0.784877845209],
            [0.87049385148, 0.85843697801, 0.833648607442, 1.59378212895, 0.687959587346, 0.687268772752, 0.686308108706, 0.687230993829],
            [1.06018898499, 1.11906265126, 1.05092914115, 3.9789654229, 0.84787549719, 0.84787549719, 0.84787549719, 0.842245933034]]

    plt.xlabel("Network")
    plt.ylabel("Energy Consumption")
    plt.title("Energy Consumption Estimation")

    dim = len(data[0])
    w = 0.75
    dimw = w / dim
    x_labels = ["2G\n     3269074 mJ\n     Original", "3G\n     596661 mJ\n     Original",
                "4G\n     370577 mJ\n     Original", "Wifi\n     95034 mJ\n     Original"]
    legend_labels = ['Level 1', 'Level 3', 'Level 6', 'Level 9', 'SVM', 'DT', 'RF', 'K-NN']
    x = np.arange(len(data))
    for i in range(len(data[0])):
        y = [d[i] for d in data]
        b = plt.bar(x + i * dimw, y, dimw, bottom=0.001, label=legend_labels[i])

    plt.legend()
    plt.ylim(0, 2)
    plt.xticks(x + dimw / 2, x_labels)
    plt.show()

def normalization(data):
    from sklearn.preprocessing import MinMaxScaler
    m = MinMaxScaler(feature_range=(0, 6))
    data = m.fit_transform(data)
    # print data[0]
    return data


def transform_dataset(data):
    # change column
    d = data.values
    tt = d[:, :-1]
    tt = normalization(tt)
    tmp = np.insert(tt, 0, d[:, -1], axis=1)
    tmp = np.insert(tmp, 1, tt[:, -1], axis=1)
    tmp = np.insert(tmp[:, :-2], 2, normalization(d[:, -3]), axis=1)
    print tmp[0, :]
    # tmp = preprocessing.scale(tmp)
    df = pd.DataFrame(tmp, columns=["CompressionLevel", "Network", "Size", "Type"])
    df.to_csv('data_visualization.csv', index=False)


if __name__ == '__main__':
    # data_plot()
    data_plot6()

    # data1 = pd.read_csv('data_visualization.csv')
    #
    # # # transfrom dataset and output the result into a new .csv
    # # transform_dataset(data1)
    #
    # # plots
    # plt.figure()
    # # parallel_coordinates(data1, 'CompressionLevel')
    # # andrews_curves(data1, 'CompressionLevel')
    # radviz(data1, 'CompressionLevel')
    # plt.show()
