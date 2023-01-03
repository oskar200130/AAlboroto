import codecs
import glob
import numpy as np
import utils
import svm
import sklearn.model_selection as sms

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

def main():
    dicc = utils.getVocabDict()
    XSpam, ySpam = readFolder(dicc, 'data_spam/spam/*.txt', 1)
    XEasy, yEasy = readFolder(dicc, 'data_spam/easy_ham/*.txt', 0)
    XHard, yHard = readFolder(dicc, 'data_spam/hard_ham/*.txt', 0)

    X = np.concatenate((XSpam, XEasy, XHard), axis=0)
    y = np.concatenate((ySpam, yEasy, yHard), axis=0)

    x_train, x_test, y_train, y_test = sms.train_test_split(X, y, test_size = 0.2, random_state = 1)
    x_train, x_val, y_train, y_val = sms.train_test_split(x_train, y_train, test_size = 0.25, random_state = 1)
    funct = svm.calcSVM(x_train, y_train, x_val, y_val)
    
    yp = funct.predict(x_test)
    cont = 0
    for i in range (len(y_test)):
        if (y_test[i] == yp[i]):
            cont += 1
    print(cont/len(y_test) * 100)


main()