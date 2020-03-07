import numpy as np
import matplotlib.pyplot as plt
from operator import itemgetter

def task_0():
    ff = np.loadtxt('train-1000-100.csv', delimiter=',')

    fiftyf = ff[:50,:]
    onehundredf = ff[:100,:]
    onefiftyf = ff[:150,:]

    fl = np.loadtxt('trainR-1000-100.csv', delimiter=',')

    fiftyl = fl[:50]
    onehundredl = fl[:100]
    onefiftyl = fl[:150]

    fft = np.loadtxt('test-1000-100.csv', delimiter=',')

    fiftyft = fft[:50, :]
    onehundredft = fft[:100, :]
    onefiftyft = fft[:150, :]

    flt = np.loadtxt('testR-1000-100.csv', delimiter=',')

    fiftylt = flt[:50]
    onehundredlt = flt[:100]
    onefiftylt = flt[:150]

    train1 = np.loadtxt('train-100-10.csv', delimiter=',')
    train2 = np.loadtxt('train-100-100.csv', delimiter=',')
    train3 = np.loadtxt('train-1000-100.csv', delimiter=',')
    train4 = np.loadtxt('train-forestfire.csv', delimiter=',')
    train5 = np.loadtxt('train-realestate.csv', delimiter=',')

    test1 = np.loadtxt('test-100-10.csv', delimiter=',')
    test2 = np.loadtxt('test-100-100.csv', delimiter=',')
    test3 = np.loadtxt('test-1000-100.csv', delimiter=',')
    test4 = np.loadtxt('test-forestfire.csv', delimiter=',')
    test5 = np.loadtxt('test-realestate.csv', delimiter=',')

    tl1 = np.loadtxt('trainR-100-10.csv', delimiter=',')
    tl2 = np.loadtxt('trainR-100-100.csv', delimiter=',')
    tl3 = np.loadtxt('trainR-1000-100.csv', delimiter=',')
    tl4 = np.loadtxt('trainR-forestfire.csv', delimiter=',')
    tl5 = np.loadtxt('trainR-realestate.csv', delimiter=',')

    test1l = np.loadtxt('testR-100-10.csv', delimiter=',')
    test2l = np.loadtxt('testR-100-100.csv', delimiter=',')
    test3l = np.loadtxt('testR-1000-100.csv', delimiter=',')
    test4l = np.loadtxt('testR-forestfire.csv', delimiter=',')
    test5l = np.loadtxt('testR-realestate.csv', delimiter=',')

    train = [[fiftyf, onehundredf, onefiftyf, train1, train2, train3, train4, train5],
             [fiftyl,onehundredl, onefiftyl, tl1, tl2, tl3, tl4, tl5]]
    test = [[fiftyft, onehundredft, onefiftyft, test1, test2, test3, test4, test5],
            [fiftylt, onehundredlt, onefiftylt, test1l, test2l, test3l, test4l, test5l]]
    return train, test

def RLS(num_features, lam, phi, y):

    I = np.identity(num_features)
    update = np.dot(np.linalg.inv(lam*I + np.dot(phi.transpose(),phi)), (np.dot(phi.transpose(),y)))

    return update

def LinearRegression(X, y, lam):

    num_features = X.shape[1]
    phi = X
    w = RLS(num_features, lam, phi, y)

    return w

def MSE(w,X,y):

    mse = 0
    num_labels = X.shape[0]
    for i in range(num_labels):

        mse += np.square(np.dot(X[i].transpose(), w) - y[i])

    return mse/num_labels

def task_1(train, test):
    print("Task 1:")
    X = train[0]
    y = train[1]

    lam = list(range(150))

    for i in range(len(X)):
        print("File", i + 1, ":")
        mse_train = []
        mse_test = []
        for l in lam:
            trainfeats = X[i]
            testfeats = test[0][i]
            testlabels = test[1][i]
            w = LinearRegression(X[i],y[i],l)

            mse_train.append((MSE(w,trainfeats,y[i]),l))
            mse_test.append((MSE(w,testfeats, testlabels), l))

        key1 = min(mse_train, key=itemgetter(0))[1]
        key2 = min(mse_test, key=itemgetter(0))[1]
        print("Optimal lambda for train:", key1, "MSE:", mse_train[key1])
        print("Optimal lambda for test:", key2, "MSE:", mse_test[key2])
        mtr = []
        mtest = []
        for m in range(len(mse_train)):
            mtr.append(mse_train[m][0])
        for m in range(len(mse_test)):
            mtest.append(mse_test[m][0])
        plot_data(i,lam,mtr,mtest)


def plot_data(i, lam, mtr, mtest):
    plt.clf()
    fig = plt.figure()
    if i < 6:
        plt.ylim(top=8)
    plt.plot(lam, mtr)
    plt.plot(lam, mtest)
    plt.legend(["train", "test"])
    plt.show()
    title = 'file' +  str(i + 1)
    #fig.savefig(title, format='png')

def task_2(train, test):
    print("\nTask 2:")

    X = train[0]
    y = train[1]
    Xt = test[0]
    yt = test[1]
    lam = k_fold_cross_val(10, train)
    print("Calculating MSE on test data:")
    for l in range(len(lam)):
        w = LinearRegression(X[l], y[l], lam[l])
        print("File", l+1, "lambda", lam[l], "test MSE:", MSE(w,Xt[l], yt[l]))


def k_fold_cross_val(k,train):
    print("Cross Validation")
    feat = train[0]
    labels = train[1]
    # lam vals 1-150
    lam = list(range(150))
    params = []

    for i in range(len(feat)):

        num_features = feat[i].shape[1]
        num_labels = feat[i].shape[0]
        data = np.c_[feat[i],labels[i]]
        np.random.shuffle(data)
        X = data[:,:num_features]
        y = data[:,num_features:]
        part = int(num_labels/k)
        lam_rec = {}

        for l in lam:
            start = 0
            mse = []
            for j in range(1,k):
                testfeat = X[start: part*j:,:]
                testlabel = y[start: part*j:,:]

                trainfeat = np.delete(X, np.s_[start: part*j], axis = 0)
                trainlabels = np.delete(y, np.s_[start: part*j], axis = 0)
                w = LinearRegression(trainfeat, trainlabels, l)
                mse.append(MSE(w,testfeat,testlabel))
                start = part*j

            ave_mse = sum(mse)/k
            lam_rec[l] = ave_mse
        key = min(lam_rec, key=lam_rec.get)
        params.append(key)
        print("File:", i+1, "optimal lambda:", key, "mse:", lam_rec[key])
    return params

def task_3(train):
    """
    Implementation of Empirical Bayes for parameter/model selection.
    :param train: a tuple of two lists. The first list are training features,
    the second are the corresponding features.
    :return: None
    """
    print("\nTask 3:\n")
    X = train[0]
    y = train[1]

    for t in range(len(X)):
        t_size = X[t].shape[1]
        num_inst = y[t].shape[0]
        gamma = 0
        a, b = [.1, 1,10], [.1,1,10]
        stop = 10**(-6)
        mse_best = float("inf")
        a_best, b_best = None, None

        for j in range(len(a)):
            for k in range(len(b)):
                dif1 = 1
                dif2 = 1
                while ((dif1 > stop) and (dif2 > stop)):
                    a_old = a[j]
                    b_old = b[k]
                    s_n = np.linalg.inv(a[j]*np.identity(t_size) + b[k]*(np.dot(X[t].T, X[t])))
                    m_n = np.dot(np.dot(b[k]*(s_n),X[t].T), y[t])
                    eig = np.linalg.eigh(np.dot(X[t].T, X[t]))[0]

                    for i in eig:
                        gamma += i/(a[j]+i)
                    a[j] = gamma/(np.dot(m_n.T, m_n))
                    b[k] = 0
                    for i in range(len(X[t])):
                        b[k] += np.square(y[t][i] - np.dot(m_n.T, X[t][i]))
                    dif1 = a_old - a[j]
                    dif2 = b_old - b[k]

                    b[k] = 1/(b[k]/(num_inst - gamma))

                #print("file ",t,":", a[j],b[k])
                mse = MSE(m_n, X[t],y[t])
                if mse < mse_best:
                    mse_best = mse
                    a_best = a[j]
                    b_best = b[k]
        print("file", t, ":", "mse:", mse_best, "a:", a_best, "b:", b_best)


#DRIVER

train, test = task_0()

task_1(train, test)

task_2(train, test)
task_3(train)