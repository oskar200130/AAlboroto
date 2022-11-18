from pickle import NONE
import numpy as np
import matplotlib.pyplot as plt
import sklearn.model_selection as sms
import sklearn.preprocessing as sp
import sklearn.linear_model as slm

def draw_data(x, y, x_i, y_i):
    plt.figure()
    plt.plot(x, y, 'bo', label = "train")
    plt.plot(x_i, y_i, 'r--', label = "y_ideal")
    plt.legend()

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

def main(): 
    x, y, x_i, y_i = gen_data(64)
    draw_data(x, y, x_i, y_i)
    x = x[:, None]
    x_train, x_test, y_train, y_test = sms.train_test_split(x, y, test_size = 0.33, random_state = 1)
    x_testO = x_test
    x_trainO = x_train

    pol = sp.PolynomialFeatures(degree= 15, include_bias=False)
    x_train = pol.fit_transform(x_train)

    scal = sp.StandardScaler()
    x_train = scal.fit_transform(x_train)

    lin = slm.LinearRegression()
    lin.fit(x_train, y_train)

    x_test = pol.transform(x_test)
    x_test = scal.transform(x_test)

    y_testpre = lin.predict(x_test)
    print(compute_cost(y_test, y_testpre))
    y_trainpre = lin.predict(x_train)
    print(compute_cost(y_train, y_trainpre))
    XData_ = np.sort(x_testO,axis=None)
    
    plt.plot(XData_, y_testpre, 'g--')
    plt.show()

main()
