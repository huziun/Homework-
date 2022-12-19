import numpy as np;
import pandas as pd;
import matplotlib.pyplot as plt;
from sklearn.model_selection import train_test_split;
import seaborn as sns;
from sklearn.metrics._scorer import r2_score;
from sklearn.preprocessing import StandardScaler;

def GetData():
    data = pd.read_csv("boston.csv")
    return data;

def Y(data):
    y = data.target;
    return y;

def X(data):
    X = data[['LSTAT']];
    return X;

def Test_Train_Split(X, y):
    X = X.to_numpy();
    X = X.reshape(-1, 1);
    y = y.to_numpy();
    X_train, X_test, Y_train, Y_test = train_test_split(X, y);
    return X_train, X_test, Y_train, Y_test;

def Corel(dataframe):
    correlation = dataframe.corr()
    heatmap = sns.heatmap(correlation, annot=True)
    heatmap.set(xlabel='Diabets values on x axis', ylabel='Diabets values on y axis\t',
                title="Correlation matrix of Diabets dataset\n")
    plt.show()
    return 0;

class Linear_Regression_1():
    def __init__(self):
        pass

    def h(self, b, w, X):
        '''
        :param b -  float or ndarry of shape [m,1], m - number of samples
        :param w - ndarray of shape [1,m],  n - number of features
        :param X - ndarray of shape [m,n], m - number of samples, n - number of features
        '''
        assert (X.shape[1] == w.shape[1])

        # YOUR_CODE. Assign expression for h to h_res
        # START_CODE
        h = b + w * X;
        h_res = h
        # END_CODE
        return h_res

def Check():
    np.random.seed(2018)
    b_check = np.random.randn()
    w_check = np.random.randn(1, 1)
    X_check = np.random.randn(10, 1)
    print('b= {}, \nw= {}, \nX= \n{}'.format(b_check, w_check, X_check))
    lin_reg_1 = Linear_Regression_1()
    print(lin_reg_1.h(b_check, w_check, X_check));
    return 0;

class Linear_Regression_2():
    '''linear regression using gradient descent
    '''

    def __init__(self):
        pass

    def J(self, h, y):
        '''
        :param h - ndarray of shape (m,1)
        :param y - ndarray of shape (m,1)
        :return expression for cost function
        '''
        if h.shape != y.shape:
            print('h.shape = {} does not match y.shape = {}.Expected {}'.format(h.shape, y.shape, (self.m, 1)))
            raise Exception('Check assertion in J')

            # YOUR_CODE. Assign expression for J to J_res
        # START_CODE
        J = 1/(2 * self.m) * np.sum((h - y) **2);
        J_res = J
        # END_CODE
        return J_res

def Check_J():
    np.random.seed(2019)
    m = 10
    y_check = np.random.randn(m, 1)
    h_check = np.random.randn(m, 1)
    print('y= {}, \nh= {}'.format(y_check, h_check))
    lin_reg_2 = Linear_Regression_2()
    lin_reg_2.m = m
    print(lin_reg_2.J(h_check, y_check));
    return 0;

class Linear_Regression_3():
    def __init__(self, max_iter=1e5, alpha=1, eps=1e-10, verbose=0):
        pass

    def h(self, b, w, X):
        '''
        :param b -  float or ndarry of shape [m,1], m - number of samples
        :param w - ndarray of shape [1,m],  n - number of features
        :param X - ndarray of shape [m,n], m - number of samples, n - number of features
        '''
        assert (X.shape[1] == w.shape[1])

        # YOUR_CODE. Insert the expression of h developed in Linear_Regression_1
        # START_CODE
        h_res = Linear_Regression_1().h(b, w, X);
        # END_CODE
        return h_res

    def J_derivative(self, params, X, y):
        '''
        :param params - tuple (b,w), where w is the 2d ndarry of shape (1,n), n- number of features
        :param X- ndarray of shape (m, n)
        :param y - ndarray of shape (m,1)
        :return tuple of derivatrives of cost function by b and w
        '''
        b, w = params
        assert (w.shape == (1, self.n))
        h_val = self.h(b, w, X)

        if h_val.shape != (self.m, 1):
            print('h.shape = {}, but expected {}'.format(h_val.shape, (self.m, 1)))
            raise Exception('Check assertion in J_derivative')

        # YOUR_CODE. Assign expressions for derivates of J by b and by w  to dJ_b and dJ_w corrrespondingly
        # START_CODE

        dJ_b = 1/self.m * np.sum((self.h(b, w, X) - y));
        dJ_w = 1/self.m * np.sum((self.h(b, w, X) - y).T@X);
        # END_CODE
        return (dJ_b, dJ_w);

def check_dj():
    np.random.seed(2020)
    m = 10
    n = 1
    X_check = np.random.randn(m, n)
    y_check = np.random.randn(m, 1)
    b_check = np.random.randn()
    w_check = np.random.randn(1, n)
    params = b_check, w_check
    print('X= {}, \ny= {}, \nb= {} \nw= {}'.format(X_check, y_check, b_check, w_check))

    lin_reg_3 = Linear_Regression_3()
    lin_reg_3.m = m
    lin_reg_3.n = n
    print(lin_reg_3.J_derivative(params, X_check, y_check))
    return 0;

class Linear_Regression_4():
    '''
    linear regression using gradient descent
    '''

    def __init__(self, max_iter=1e5, alpha=0.01, eps=1e-10, verbose=0):
        '''
        :param verbose: set 1 to display more details of J val changes
        '''
        self.max_iter = max_iter
        self.alpha = alpha
        self.eps = eps
        self.verbose = verbose

    def h(self, b, w, X):
        '''
        :param b -  float or ndarry of shape [m,1], m - number of samples
        :param w - ndarray of shape [1,m],  n - number of features
        :param X - ndarray of shape [m,n], m - number of samples, n - number of features
        '''
        assert (X.shape[1] == w.shape[1])

        # YOUR_CODE. Insert the expression of h developed in Linear_Regression_1
        # START_CODE
        h_res = Linear_Regression_1().h(b, w, X);
        # END_CODE

        if h_res.shape != (X.shape[0], 1):
            print('h.shape = {} but expected {}'.format(h_res.shape, (self.m, 1)))
            raise Exception('Check assertion in h')

        return h_res

    def J(self, h, y):
        '''
        :param h - ndarray of shape (m,1)
        :param y - ndarray of shape (m,1)
        :return expression for cost function
        '''
        if h.shape != y.shape:
            print('h.shape = {} does not match y.shape = {}.Expected {}'.format(h.shape, y.shape, (self.m, 1)))
            raise Exception('Check assertion in J')
            # YOUR_CODE. Insert the expression of J developed in Linear_Regression_2
        # START_CODE
        j = Linear_Regression_2();
        j.m = self.m;
        J_res = j.J(h, y);
        # END_CODE
        return J_res

    def J_derivative(self, params, X, y):
        '''
        :param params - tuple (b,w), where w is the 2d ndarry of shape (1,n), n- number of features
        :param X- ndarray of shape (m, n)
        :param y - ndarray of shape (m,1)
        :return tuple of derivatrives of cost function by b and w
        '''

        b, w = params
        assert (w.shape == (1, self.n))
        h_val = self.h(b, w, X);

        if h_val.shape != (self.m, 1):
            print('h.shape = {}, but expected {}'.format(h_val.shape, (self.m, 1)))
            raise Exception('Check assertion in J_derivative')

        # YOUR_CODE. Insert the expressions for derivates of J by b and by w to dJ_b and dJ_w developed in Linear_Regression_3
        # START_CODE
        LR3 = Linear_Regression_3();
        LR3.m = self.m;
        LR3.n = self.n;
        jb, jx = LR3.J_derivative(params, X, y);
        dJ_b = jb;
        dJ_w = jx;
        # END_CODE
        return (dJ_b, dJ_w)

    def fit(self, X, y):
        '''
        :param X - ndarray training set of shape [m,n], m - number of samples, n - number of features
        :param y - ndarray - 1d array
        :return: True in case of successful fit
        '''

        if self.verbose:
            print('Running gradient descent with alpha = {}, eps= {}, max_iter= {}'.format(
                self.alpha, self.eps, self.max_iter))

        self.m, self.n = X.shape  # number of samples, number of features
        y = y.reshape(self.m, 1)  # make it 2 d to make sure it corresponds to h_val
        b = 0  # init intercept with 0
        w = np.zeros(self.n).reshape(1, -1)  # make sure it's shape is [1,n]
        params = (b, w)

        self.J_hist = [-1]  # used for keeping J values. Init with -1 to avoid 0 at first iter
        continue_iter = True  # flag to continue next iter (grad desc step)
        iter_number = 0  # used for limit by max_iter

        while continue_iter:
            # Do step of gradient descent
            # YOUR_CODE. Develop one step of gradien descent
            # START_CODE
            dJ_b, dJ_w = self.J_derivative(params, X, y);
            b = b - self.alpha * dJ_b;
            w = w - self.alpha * dJ_w;
            params = (b, w);
            # END_CODE

            # keep history of J values
            self.J_hist.append(self.J(self.h(b, w, X), y))
            if self.verbose:
                print('b = {}, w= {}, J= {}'.format(b, w, self.J_hist[-1]))
            # check criteria of exit the loop (finish grad desc)
            if self.max_iter and iter_number > self.max_iter:  # if max_iter is provided and limit succeeded
                continue_iter = False
            elif np.abs(self.J_hist[iter_number - 1] - self.J_hist[iter_number]) < self.eps:  # if accuracy is succeeded
                continue_iter = False
            iter_number += 1

        # store the final params to further using
        self.intercept_, self.coef_ = params
        return True

def Check_Gradient():
    np.random.seed(2021)
    m = 10
    n = 1
    X_check = np.random.randn(m, n)
    y_check = np.random.randn(m, 1)
    print('X= {}, \ny= {}'.format(X_check, y_check))
    lin_reg_4 = Linear_Regression_4(alpha=1, max_iter=5, verbose=1)
    print(lin_reg_4.fit(X_check, y_check))
    return 0;

class Linear_Regression():
    '''
    linear regression using gradient descent
    '''

    def __init__(self, max_iter=1e5, alpha=0.01, eps=1e-10, verbose=0):
        '''
        :param verbose: set 1 to display more details of J val changes
        '''
        self.max_iter = max_iter
        self.alpha = alpha
        self.eps = eps
        self.verbose = verbose

    def h(self, b, w, X):
        '''
        :param b -  float or ndarry of shape [m,1], m - number of samples
        :param w - ndarray of shape [1,m],  n - number of features
        :param X - ndarray of shape [m,n], m - number of samples, n - number of features
        '''
        assert (X.shape[1] == w.shape[1])

        # YOUR_CODE. Insert the expression of h developed in Linear_Regression_1
        # START_CODE
        h_res = b + w * X
        # END_CODE

        if h_res.shape != (X.shape[0], 1):
            print('h.shape = {} but expected {}'.format(h_res.shape, (self.m, 1)))
            raise Exception('Check assertion in h')
        return h_res

    def J(self, h, y):
        '''
        :param h - ndarray of shape (m,1)
        :param y - ndarray of shape (m,1)
        :return expression for cost function
        '''
        if h.shape != y.shape:
            print('h.shape = {} does not match y.shape = {}.Expected {}'.format(h.shape, y.shape, (self.m, 1)))
            raise Exception('Check assertion in J')
            # YOUR_CODE. Insert the expression of J developed in Linear_Regression_2
        # START_CODE
        J_res = 1/(2 * self.m) * np.sum((h - y) **2)
        # END_CODE

        return J_res

    def J_derivative(self, params, X, y):
        '''
        :param params - tuple (b,w), where w is the 2d ndarry of shape (1,n), n- number of features
        :param X- ndarray of shape (m, n)
        :param y - ndarray of shape (m,1)
        :return tuple of derivatrives of cost function by b and w
        '''

        b, w = params
        assert (w.shape == (1, self.n))
        h_val = self.h(b, w, X)
        if h_val.shape != (self.m, 1):
            print('h.shape = {}, but expected {}'.format(h_val.shape, (self.m, 1)))
            raise Exception('Check assertion in J_derivative')

        # YOUR_CODE. Insert the expressions for derivates of J by b and by w to dJ_b and dJ_w developed in Linear_Regression_3
        # START_CODE
        dJ_b = 1/self.m * np.sum((self.h(b, w, X) - y));
        dJ_w = 1/self.m * np.sum((self.h(b, w, X) - y).T@X);
        # END_CODE

        return (dJ_b, dJ_w)

    def fit(self, X, y):
        '''
        :param X - ndarray training set of shape [m,n], m - number of samples, n - number of features
        :param y - ndarray - 1d array
        :return: True in case of successful fit
        '''
        if self.verbose:
            print('Running gradient descent with alpha = {}, eps= {}, max_iter= {}'.format(
                self.alpha, self.eps, self.max_iter))
        self.m, self.n = X.shape  # number of samples, number of features
        y = y.reshape(self.m, 1)  # make it 2 d to make sure it corresponds to h_val
        b = 0  # init intercept with 0
        w = np.zeros(self.n).reshape(1, -1)  # make sure it's shape is [1,n]
        params = (b, w)

        self.J_hist = [-1]  # used for keeping J values. Init with -1 to avoid 0 at first iter
        continue_iter = True  # flag to continue next iter (grad desc step)
        iter_number = 0  # used for limit by max_iter

        while continue_iter:
            # Do step of gradient descent
            # YOUR_CODE. Insert one step of gradien descent developed in Linear_Regression_4
            # START_CODE
            dJ_b, dJ_w = self.J_derivative(params, X, y);
            b = b - self.alpha * dJ_b;
            w = w - self.alpha * dJ_w;
            params = (b, w);
            # END_CODE

            # keep history of J values
            self.J_hist.append(self.J(self.h(b, w, X), y))
            if self.verbose:
                print('b = {}, w= {}, J= {}'.format(b, w, self.J_hist[-1]))
            # check criteria of exit the loop (finish grad desc)
            if self.max_iter and iter_number > self.max_iter:  # if max_iter is provided and limit succeeded
                continue_iter = False
            elif np.abs(self.J_hist[iter_number - 1] - self.J_hist[iter_number]) < self.eps:  # if accuracy is succeeded
                continue_iter = False
            iter_number += 1

        # store the final params to further using
        self.intercept_, self.coef_ = params
        return True

    def draw_cost_changes(self):
        J_hist = self.J_hist[1:]
        plt.figure()
        plt.scatter(np.arange(0, len(J_hist)), J_hist, s=20, marker='.', c='b')
        plt.xlabel('Iterations')
        plt.ylabel('Cost function J value')
        title_str = 'Complited: {}, alpha ={}, max_iter={}, eps={}'.format(len(self.J_hist) - 2, self.alpha,
                                                                           self.max_iter, self.eps)
        # Note: len(J_hist)-2) due to first one is -1 (was not iteration), iter + 1  at the end  of the gradient loop
        plt.title(title_str)

    def predict(self, X):
        '''
        :param X - ndarray of shape (?,n)
        :return
        '''
        return self.h(self.intercept_, self.coef_, X)

    def score(self, X_test, y_test):
        '''
        :param X_test - ndarray testing set or any for prediction of shape [?,n], ? - number of samples, n - number of features
        :param y_test - ndarray - 1d array
        :return R2 score of y_test and prediction for X_test
        '''
        z = self.predict(X_test)
        from sklearn.metrics._scorer import r2_score
        return (r2_score(y_test, z))

def Check_LR_for_realData(X_train, y_train, X_test, y_test):
    print('X_train.shape= ', X_train.shape)
    print('y_train.shape= ', y_train.shape)

    #print('X_train= \n{}'.format(X_train[:5, :]))
    lin_reg = Linear_Regression(alpha=0.01, verbose=0, eps=1e-8)
    lin_reg.fit(X_train, y_train)
    lin_reg.draw_cost_changes()
    print('R2 Score =', lin_reg.score(X_test, y_test))
    print('b: {}, w= {}'.format(lin_reg.intercept_, lin_reg.coef_))
    return lin_reg;

def Draw(X_train, y_train, lin_reg):
    if X_train.shape[1] > 1:
        raise Exception('Select single feature to plot')
    plt.figure()
    plt.scatter(X_train, y_train)
    x_line = np.array([np.min(X_train), np.max(X_train)])
    z_line = lin_reg.predict(x_line.reshape(-1, 1))
    plt.plot(x_line, z_line, '-', c='red');
    return 0;

def Start():
    data = GetData();
    #Corel(data);
    x = X(data);

    y = Y(data);

    X_train, X_test, Y_train, Y_test = Test_Train_Split(x, y);

    scaler = StandardScaler();
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    print("~~Check~~");
    Check();
    print("~~Check_J~~");
    Check_J();
    print("~~Check_dj~~");
    check_dj();
    print("~~Check_Gradient~~");
    Check_Gradient();
    lin_reg = Check_LR_for_realData(X_train, Y_train, X_test, Y_test);
    Draw(X_train, Y_train, lin_reg);
    plt.show();
    return 0;

Start()