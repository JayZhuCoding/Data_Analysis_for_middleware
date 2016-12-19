#/usr/bin/python
import numpy as np
import pickle as pickle
import random as random
from sklearn.svm import SVC
from time import time
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler
from sklearn import preprocessing
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_predict
from sklearn.model_selection import cross_val_score
from sklearn import metrics
from sklearn.externals import joblib



def Data_preprocess(Network=2):
    # Data Generation:
    # The following lines of code help to gather the
    # information needed from the original dataset,
    # and to generate a usable dataset for data analysis.
    switcher = {
                2 : "dataset1_txt_2g.txt",
                3 : "dataset1_txt_3g.txt",
                4 : "dataset1_txt_4g.txt",
                5 : "dataset1_txt_wifi.txt"
    }
    dataSet = switcher.get(Network)
    # Analyze data
    rst = Data_Analysis(dataSet)

    # transfrom data
    Data_Transfrom(dataSet, Network)

    # Generate final dataset
    DataSet_Generation(rst)

def _file_type(features):
    if ".txt" in features[1].replace(" datasize", ""):
        return 1
    if ".jpg" in features[1].replace(" datasize", ""):
        return 2
    if ".jpeg" in features[1].replace(" datasize", ""):
        return 2
    if ".gif" in features[1].replace(" datasize", ""):
        return 3
    if ".png" in features[1].replace(" datasize", ""):
        return 4
    if ".bmp" in features[1].replace(" datasize", ""):
        return 5
    # print features[1].replace("datasize", "")

def _file_size(features):
    return features[2].replace(" network3g", "").strip()

def _opt_method(features):
    return features[4].replace("def", "").strip()

def _network(Network):
    return Network

def Data_Analysis(dataSet):
    prev = 0
    total = 0
    i = 0
    n = 142
    with open(dataSet, "r") as ins:
        List = []
        for line in ins:
            string = ",".join("".join(line.split(",")[-1]).split(":")[-2:]).replace(" Total: ", "")
            energy = string.split(",")
            total_energy = energy[1].strip()
            total += int(total_energy)
            List.append(int(total_energy)-prev)
            # print ss-prev
            prev = int(total_energy)
            i += 1
            # print i

    ins.close()
    rst = []
    deviation = 0
    sum1 = 0
    sum2 = 0
    sum3 = 0
    sum4 = 0
    sum5 = 0
    for j in range(1, n):
        min_value = min(List[j], List[j + n], List[j + n*2], List[j + n*3], List[j + n*4])
        # print min_value - List[j]
        deviation -= min_value - List[j]

        sum1 += List[j]
        sum2 += List[j + n]
        sum3 += List[j + n*2]
        sum4 += List[j + n*3]
        sum5 += List[j + n*4]

        if (min_value - List[j]) == 0:
            rst.append(0)
            continue
        if min_value == List[j + n]:
            rst.append(n)
            continue
        if min_value == List[j + n*2]:
            rst.append(n*2)
            continue
        if min_value == List[j + n*3]:
            rst.append(n*3)
            continue
        if min_value == List[j + n*4]:
            rst.append(n*4)
            continue

    print sum1
    print sum2
    print sum3
    print sum4
    print sum5
    print deviation

    print "********************* Non-Adaptive Optimization Methods ***********************"
    print "Opt. Compression level 1:", 100 - (float(sum2*100)/sum1), "%"
    print "Opt. Compression level 3:", 100 - (float(sum3*100)/sum1), "%"
    print "Opt. Compression level 6:", 100 - (float(sum4*100)/sum1), "%"
    print "Opt. Compression level 9:", 100 - (float(sum5*100)/sum1), "%"

    print "Maximum Optimization Percentage:", float(deviation*100)/sum1, "%"
    # print "Comparison: ", ((float(sum2)/sum1)-1)*100
    print "Number of sample:", len(rst)

    return rst

def Data_Transfrom(dataSet, Network):
    dataset_orig = open("dataset_orig.txt", "w")
    with open(dataSet, "r") as ins:
        # List = []
        for line in ins:
            string = ",".join("".join(line.split(",")[0:-1]).split(":")[:]).replace(" Total: ", "")
            features = string.split(",")
            dataset_orig.write(str(_file_type(features)) + "," + str(_file_size(features))
                               + "," + str(_network(Network)) + "," + str(_opt_method(features)) + "\n")

    ins.close()
    dataset_orig.close()

def DataSet_Generation(rst):
    dataset = open("dataset_tmp.txt", "w")
    with open("dataset_orig.txt", "r") as ins:
        List = []
        for line in ins:
            List.append(line)
    zeros = 0
    for i in range(len(rst)):
        if rst[i] == 0:
            zeros += 1
        dataset.write(List[i + rst[i] + 1])

    print "No compression samples:", zeros, "\n"
    ins.close()
    dataset.close()

def Get_random_data():
    with open('dataset_random.txt', 'r') as source:
        data = [(random.random(), line) for line in source]
    data.sort()

    with open('dataset_random.txt', 'w') as target:
        for _, line in data:
            target.write(line)

    source.close()
    target.close()

def Get_Samples(dataset = "dataset_random.txt", Test_Size=0.5):
    data = np.loadtxt(open(dataset, "rb"), delimiter=",")
    #print data.shape
    y = data.transpose()[3][:]
    y = y.transpose()
    X = data.transpose()
    X = X[0:3][:].transpose()
    # print X.shape, y .shape
    # n = len(X)

    #Standardization
    # X = StandardScaler(with_std=False).fit_transform(X)
    # X = preprocessing.scale(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=Test_Size, random_state=100)
    print "Train-set shape:", X_train.shape, y_train.shape
    print "Test-set shape:", X_test.shape, y_test.shape

    return X_train, y_train, X_test, y_test

def _calculate_energy_saving(clf, dataSet, Network_Condition):
    prev = 0
    total = 0
    i = 0
    n = 142
    switcher = {
                2 : "dataset1_txt_2g.txt",
                3 : "dataset1_txt_3g.txt",
                4 : "dataset1_txt_4g.txt",
                5 : "dataset1_txt_wifi.txt"
    }
    Network_Condition = switcher.get(Network_Condition)
    data = np.loadtxt(open(dataSet, "rb"), delimiter=",")
    X = data.transpose()
    X = X[0:3][:].transpose()

    with open(Network_Condition, "r") as ins:
        List = []
        for line in ins:
            string = ",".join("".join(line.split(",")[-1]).split(":")[-2:]).replace(" Total: ", "")
            energy = string.split(",")
            total_energy = energy[1].strip()
            total += int(total_energy)
            List.append(int(total_energy)-prev)
            # print ss-prev
            prev = int(total_energy)
            i += 1
            # print i
    ins.close()
    optimization = 0
    sum_no_opt = 0
    opt_map = {
        0 : 0,
        1 : 1,
        3 : 2,
        6 : 3,
        9 : 4
    }

    for j in range(1, n):
        optimization += List[j + (opt_map.get(clf.predict(X)[j-1])*n)]
        sum_no_opt += List[j]

    print "Before optimization:", sum_no_opt, "mj"
    print "After optimization:", optimization, "mj"

    return float(optimization*100)/sum_no_opt


class Algorithm(object):

    def __init__(self, X_train, y_train, X_test, y_test, Network_Condition):
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        self.Network_Condition = Network_Condition

    # SVM
    def S_Accuracy_SVM(self):
        # C_range = np.logspace(-6, 6, 13)
        # gamma_range = np.logspace(-9, 3, 13)
        # param_grid = dict(gamma=gamma_range, C=C_range)
        # cv = StratifiedShuffleSplit(n_splits=5, test_size=0.2, random_state=5)
        # grid = GridSearchCV(SVC(), param_grid=param_grid, cv=cv, n_jobs=-1)
        # grid.fit(self.X_train, self.y_train)
        #
        # print("The best parameters are %s with a score of %0.2f"
        #       % (grid.best_params_, grid.best_score_))

        clf = SVC(kernel='rbf', gamma=1000.0, C=100.0)
        clf.fit(self.X_train, self.y_train)

        print "****** S Results ******"
        startTime = time()
        clf.predict(self.X_test)
        print "Time Spent:", (time() - startTime)*1000, "ms"
        SVM_clf_persistence = pickle.dumps(clf)
        print "S Accuracy is:", clf.score(self.X_test, self.y_test), "\n"

        X_train_1, y_train_1, X_test_1, y_test_1 = Get_Samples(dataset="dataset_tmp.txt", Test_Size=0.999)
        # print np.equal(clf.predict(X_test_1), y_test_1)
        print "S Accuracy on Original Dataset is:", clf.score(X_test_1, y_test_1), "\n"
        energy_saving = _calculate_energy_saving(clf, "dataset_tmp.txt", Network_Condition)
        print "Energy Consumption(Optimized):", energy_saving, "%\n"
        return energy_saving

    # Neural Network
    def NN_Accuracy(self):
        clf = MLPClassifier(activation="tanh", solver='sgd',
                            alpha=1e-8, hidden_layer_sizes = (16,16), random_state = None)
        clf.fit(self.X_train, self.y_train)

        print "****** NN Results ******"
        startTime = time()
        clf.predict(self.X_test)
        print "Time Spent:", (time() - startTime)*1000, "ms"
        print "NN Accuracy is:", clf.score(self.X_test, self.y_test), "\n"

        X_train_1, y_train_1, X_test_1, y_test_1 = Get_Samples(dataset="dataset_tmp.txt", Test_Size=0.999)
        print "NN Accuracy on Original Dataset is:", clf.score(X_test_1, y_test_1), "\n"
        energy_saving = _calculate_energy_saving(clf, "dataset_tmp.txt", Network_Condition)
        print "Energy Consumption(Optimized):", energy_saving, "%\n"
        return energy_saving

    # Random Forest
    def RF_Accuracy(self):
        clf = RandomForestClassifier(n_estimators=80, n_jobs=-1)
        clf.fit(self.X_train, self.y_train)

        print "****** RF Results ******"
        startTime = time()
        clf.predict(self.X_test)
        print "Time Spent:", (time() - startTime)*1000, "ms"
        print "RF Accuracy is:", clf.score(self.X_test, self.y_test), "\n"

        X_train_1, y_train_1, X_test_1, y_test_1 = Get_Samples(dataset="dataset_tmp.txt", Test_Size=0.999)
        print "RF Accuracy on Original Dataset is:", clf.score(X_test_1, y_test_1), "\n"
        energy_saving = _calculate_energy_saving(clf, "dataset_tmp.txt", Network_Condition)
        print "Energy Consumption(Optimized):", energy_saving, "%\n"
        return energy_saving

    #k-NN
    def KN_Accuracy(self):
        clf = KNeighborsClassifier(n_neighbors=1)
        clf.fit(self.X_train, self.y_train)

        print "****** KN Results ******"
        startTime = time()
        clf.predict(self.X_test)
        print "Time Spent:", (time() - startTime)*1000, "ms"
        print "KN Accuracy is:", clf.score(self.X_test, self.y_test), "\n"

        X_train_1, y_train_1, X_test_1, y_test_1 = Get_Samples(dataset="dataset_tmp.txt", Test_Size=0.999)
        print "KN Accuracy on Original Dataset is:", clf.score(X_test_1, y_test_1), "\n"
        energy_saving = _calculate_energy_saving(clf, "dataset_tmp.txt", Network_Condition)
        print "Energy Consumption(Optimized):", energy_saving, "%\n"
        return energy_saving

    # Naive Bayes
    def NB_Accuracy(self):
        clf = GaussianNB()
        clf.fit(self.X_train, self.y_train)

        print "****** NB Results ******"
        startTime = time()
        clf.predict(self.X_test)
        print "Time Spent:", (time() - startTime)*1000, "ms"
        print "NB Accuracy is:", clf.score(self.X_test, self.y_test), "\n"

        X_train_1, y_train_1, X_test_1, y_test_1 = Get_Samples(dataset="dataset_tmp.txt", Test_Size=0.999)
        print "NB Accuracy on Original Dataset is:", clf.score(X_test_1, y_test_1), "\n"
        energy_saving = _calculate_energy_saving(clf, "dataset_tmp.txt", Network_Condition)
        print "Energy Consumption(Optimized):", energy_saving, "%\n"
        return energy_saving

    # Decision Tree
    def DT_Accuracy(self):
        # clf = DecisionTreeClassifier(random_state=2)
        # clf.fit(self.X_train, self.y_train)
        # # Model Persistence
        # joblib.dump(clf, 'DecisionTreeModel.pkl')
        #
        # print "****** DT Results ******"
        # startTime = time()
        # clf.predict(self.X_test)
        # print "Time Spent:", (time() - startTime)*1000, "ms"
        # # DT_clf_persistence = pickle.dumps(clf)
        clf = joblib.load('./model/DecisionTreeModel.pkl')
        print "DT Accuracy is:", clf.score(self.X_test, self.y_test), "\n"

        X_train_1, y_train_1, X_test_1, y_test_1 = Get_Samples(dataset="dataset_tmp.txt", Test_Size=0.999)
        print "DT Accuracy on Original Dataset is:", clf.score(X_test_1, y_test_1), "\n"
        energy_saving = _calculate_energy_saving(clf, "dataset_tmp.txt", Network_Condition)
        print "Energy Consumption(Optimized):", energy_saving, "%\n"
        return energy_saving


def K_Fold(X_train, y_train, k):
    kf = KFold(n_splits=k)
    clf_SVM = SVC(kernel='rbf', gamma=1000.0, C=100.0)
    clf_DT = DecisionTreeClassifier(random_state=2)
    clf_RF = RandomForestClassifier(n_estimators=80, n_jobs=-1)

    print cross_val_score(clf_SVM, X_train, y_train, cv=kf, n_jobs=-1)
    predicted = cross_val_predict(clf_SVM, X_train, y_train, cv=kf, n_jobs=-1)
    print metrics.accuracy_score(y_train, predicted)

    print cross_val_score(clf_DT, X_train, y_train, cv=kf, n_jobs=-1)
    predicted = cross_val_predict(clf_DT, X_train, y_train, cv=kf, n_jobs=-1)
    print metrics.accuracy_score(y_train, predicted)

    print cross_val_score(clf_RF, X_train, y_train, cv=kf, n_jobs=-1)
    predicted = cross_val_predict(clf_RF, X_train, y_train, cv=kf, n_jobs=-1)
    print metrics.accuracy_score(y_train, predicted)

def energy_optimization(X_train, y_train, X_test, y_test, Network_Condition, iteration=1):
    network_map = {
                2 : "2G Network",
                3 : "3G Network",
                4 : "4G Network",
                5 : "Wifi Network"
    }

    al = Algorithm(X_train, y_train, X_test, y_test, Network_Condition)

    avg_S = 0
    avg_RF = 0
    avg_KN = 0
    avg_DT = 0
    avg_NN = 0
    avg_NB = 0

    for i in range(iteration):
        avg_S += al.S_Accuracy_SVM()
        # avg_NN += al.NN_Accuracy()
        avg_RF += al.RF_Accuracy()
        avg_KN += al.KN_Accuracy()
        # avg_NB += al.NB_Accuracy()
        avg_DT += al.DT_Accuracy()

    print "Network:", network_map.get(Network_Condition)
    print "Average SVM optimization:", float(100*iteration - avg_S)/iteration, "%"
    print "Average Random Forest optimization:", float(100*iteration - avg_RF)/iteration, "%"
    print "Average K-NN optimization:", float(100*iteration - avg_KN)/iteration, "%"
    print "Average Decision Tree optimization:", float(100*iteration - avg_DT)/iteration, "%"
    # print "Average NN optimization:", float(100 * iteration - avg_NN) / iteration, "%"
    # print "Average NB optimization:", float(100 * iteration - avg_NB) / iteration, "%"


if __name__=="__main__":
    # Network Condition Values
    # network_map = {
    #             2 : "2G Network",
    #             3 : "3G Network",
    #             4 : "4G Network",
    #             5 : "Wifi Network"
    # }
    Network_Condition = 3
    iteration = 1
    Test_Size = 0.5

    # Pre-process data: used to pre-process data and generate samples in tmp dataset
    Data_preprocess(Network = Network_Condition)

    # # Randomly assign line numbers to samples: reassign samples in random dataset
    # Get_random_data()

    # Get train and test datasets
    X_train, y_train, X_test, y_test = Get_Samples(dataset="dataset_random.txt", Test_Size=Test_Size)

    # # K Fold Cross-Validation
    # K_Fold(X_train, y_train, 20)

    #Prediction
    energy_optimization(X_train, y_train, X_test, y_test, Network_Condition, iteration)
