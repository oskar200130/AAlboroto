from pickle import NONE
import numpy as np
import matplotlib.pyplot as plt
import sklearn.model_selection as sms
import sklearn.preprocessing as sp
import sklearn.linear_model as slm

def draw_data(x, y, x_i, y_i, x_tr, y_tr):
    plt.figure()
    plt.plot(x, y, 'bo', label = "train")
    plt.plot(x_i, y_i, c = 'red', label = "y_ideal", linestyle='dashed')
    plt.plot(x_tr, y_tr, c='green', label='predict', linestyle='dashed', linewidth=2, markersize=12)

    plt.legend()
    plt.show()

def gen_data(m, seed=1, scale=0.7):
    """ generate a data set based on a x^2 with added noise """
    c = 0
    x_train = np.linspace(0, 49, m)
    np.random.seed(seed)
    y_ideal = x_train**2 + c
    y_train = y_ideal + scale * y_ideal*(np.random.sample((m,))-0.5)
    x_ideal = x_train
    return x_train, y_train, x_ideal, y_ideal

def compute_cost(y, y_predict):
    sum = 0
    m = len(y)

    sum = np.sum((y_predict - y) ** 2)
    
    return sum/(2*m)

def train(degr, x_train, y_train):
    pol = sp.PolynomialFeatures(degree= degr, include_bias=False)
    x_train = pol.fit_transform(x_train)

    scal = sp.StandardScaler()
    x_train = scal.fit_transform(x_train)

    lin = slm.LinearRegression()
    lin.fit(x_train, y_train)

    return pol, scal, lin, x_train

def test(x_test, y_test, x_train, y_train, pol, scal, lin):
    x_test = pol.transform(x_test)
    x_test = scal.transform(x_test)

    y_testpre = lin.predict(x_test)
    test_cst = compute_cost(y_test, y_testpre)
    print(test_cst)
    y_trainpre = lin.predict(x_train)
    train_cst = compute_cost(y_train, y_trainpre)
    #print(train_cst)

    return test_cst, train_cst, y_testpre

def sobreAjuste(x, y, x_i, y_i):
    x = x[:, None]
    x_train, x_test, y_train, y_test = sms.train_test_split(x, y, test_size = 0.33, random_state = 1)
    x_testO = x_test

    pol, scal, lin, x_train = train(15, x_train, y_train)

    tc, trc, y_testpre = test(x_test, y_test, x_train, y_train, pol, scal, lin)
    
    y_sorted = [y for _,y in sorted(zip(x_testO, y_testpre))]
    x_sorted = np.sort(x_testO, axis=None)

    draw_data(x, y, x_i, y_i, x_sorted, y_sorted)

def main(): 
    x, y, x_i, y_i = gen_data(64)
    #sobreAjuste(x, y, x_i, y_i)

    x = x[:, None]
    x_train, x_test, y_train, y_test = sms.train_test_split(x, y, test_size = 0.2, random_state = 1)
    x_train, x_val, y_train, y_val = sms.train_test_split(x_train, y_train, test_size = 0.25, random_state = 1)
    x_testO = x_test

    minCos = 0
    degree = 0

    for i in range(10):
        pol, scal, lin, x_train_t = train(i + 1, x_train, y_train)
        test_cst, train_cst, y_tspr = test(x_val, y_val, x_train_t, y_train, pol, scal, lin)
        if(minCos == 0 or test_cst < minCos):
            minCos = test_cst
            degree = i+1
            print(i+1)
    
    pol, scal, lin, x_train = train(degree, x_train, y_train)
    tc, trc, y_testpre = test(x_test, y_test, x_train, y_train, pol, scal, lin)

    y_sorted = [y for _,y in sorted(zip(x_testO, y_testpre))]
    x_sorted = np.sort(x_testO, axis=None)

    draw_data(x, y, x_i, y_i, x_sorted, y_sorted)
    
main()