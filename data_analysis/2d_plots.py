import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from pandas.tools.plotting import parallel_coordinates
from pandas.tools.plotting import andrews_curves
from pandas.tools.plotting import radviz
from sklearn import preprocessing

matplotlib.style.use('ggplot')
# #
# data = [[11.2676056338, 21.8309859155, 33.8028169014, 21.1267605634, 11.2676056338],
#         [30.2816901408, 33.0985915493, 15.4929577465, 15.4929577465, 4.92957746479],
#         [24.6478873239, 26.7605633803, 20.4225352113, 24.6478873239, 2.81690140845],
#         [45.7746478873, 21.8309859155, 10.5633802817, 18.3098591549, 2.81690140845]]
#
# df2 = pd.DataFrame(np.asarray(data),
#                    columns=['Level 0', 'Level 1',
#                             'Level 3', 'Level 6', 'Level 9', ], index=['2G', '3G', '4G', 'Wifi'])
#
# plt.figure()
# plt.xlabel('Network Condition')
# plt.ylabel('Configuration Percentage')
# df2.plot.bar()
# plt.show()

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
    data1 = pd.read_csv('data_visualization.csv')

    # # transfrom dataset and output the result into a new .csv
    # transform_dataset(data1)

    # plots
    plt.figure()
    # parallel_coordinates(data1, 'CompressionLevel')
    # andrews_curves(data1, 'CompressionLevel')
    radviz(data1, 'CompressionLevel')
    plt.show()
