import codecs
import glob
import numpy as np
import utils
import svm
import sklearn.model_selection as sms
import logistic_reg as lr
import nn


def readMail(path, dicc):
    vec = np.zeros(len(dicc))
    email_contents = codecs.open(path, 'r', encoding='utf-8', errors='ignore').read()
    email = utils.email2TokenList(email_contents)
    for i in range (len(email)):
        try:
            num = dicc[email[i]]
            vec[num] = 1
        except: 
            continue
    return vec

def readFolder(dicc, folderPath, isSpam):
    docs = glob.glob(folderPath)
    X = np.zeros((len(docs), len(dicc)))
    y = np.full((len(docs)), isSpam)
    for i in range (len(docs)):
        X[i] = readMail(docs[i], dicc)

    return X, y

def withSVM(X, y):
    x_train, x_test, y_train, y_test = sms.train_test_split(X, y, test_size = 0.2, random_state = 1)
    x_train, x_val, y_train, y_val = sms.train_test_split(x_train, y_train, test_size = 0.25, random_state = 1)

    funct = svm.calcSVM(x_train, y_train, x_val, y_val)
    
    yp = funct.predict(x_test)
    cont = 0
    for i in range (len(y_test)):
        if (y_test[i] == yp[i]):
            cont += 1
    print(cont/len(y_test) * 100)

def withLogisticRegresion(X, y):
    x_train, x_test, y_train, y_test = sms.train_test_split(X, y, test_size = 0.2, random_state = 1)
    x_train, x_val, y_train, y_val = sms.train_test_split(x_train, y_train, test_size = 0.25, random_state = 1)
    lambd = [0.000001, 0.00001, 0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 300, 600, 900]

    bestLamb = -1
    acierto = -1
    bestW = np.zeros(len(x_train[0]))
    bestB = 0
    
    for l in lambd: 
        
        aciertoAux, wAux, bAux = lr.calcBest(x_train, y_train, x_val, y_val, l, 200)

        if (bestLamb == -1 or acierto < aciertoAux):
            bestLamb = l
            acierto = aciertoAux
            bestW = wAux
            bestB = bAux

    print(lr.test(x_test, y_test, bestW, bestB))
    
def withNN(X, y):
    x_train, x_test, y_train, y_test = sms.train_test_split(X, y, test_size = 0.2, random_state = 1)
    x_train, x_val, y_train, y_val = sms.train_test_split(x_train, y_train, test_size = 0.25, random_state = 1)

    y_hot = np.zeros([len(y_train), 2])
    for i in range(len(y_train)):
        y_hot[i][y[i]] = 1
    lambd = [0.000001, 0.00001, 0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 300, 600, 900]

    theta1 = np.random.random((25, len(x_train[0]) + 1)) * (2*0.12)  - 0.12
    theta2 = np.random.random((2, 26))  * (2*0.12)  - 0.12
    arr = np.concatenate([theta1.ravel(), theta2.ravel()])

    bestLamb = -1
    acierto = -1

    for l in lambd:
        aciertoAux,theta1Aux, theta2Aux = nn.calcBest(x_train, y_hot, x_val, y_val, arr, l, 200)
        if (bestLamb == -1 or aciertoAux > acierto):
            bestLamb = l
            acierto = aciertoAux 
            theta1 = theta1Aux
            theta2 = theta2Aux

    print(nn.test(x_test, y_test, theta1, theta2))


def main():
    dicc = utils.getVocabDict()
    XSpam, ySpam = readFolder(dicc, 'data_spam/spam/*.txt', 1)
    XEasy, yEasy = readFolder(dicc, 'data_spam/easy_ham/*.txt', 0)
    XHard, yHard = readFolder(dicc, 'data_spam/hard_ham/*.txt', 0)

    X = np.concatenate((XSpam, XEasy, XHard), axis=0)
    y = np.concatenate((ySpam, yEasy, yHard), axis=0)

    withLogisticRegresion(X, y)



main()