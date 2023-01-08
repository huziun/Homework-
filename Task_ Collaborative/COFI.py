#from collaborative_filtering import fit_collaborative_filtering
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def GetData():
    data = pd.read_excel('ratings_movies.xlsx', header = None);
    data = np.array(data)
    return data

def J(Y, R, X, Theta, lambd):
    '''
    params: 1d vector  of X and Theta
    :return expression for cost function
    '''

    assert (X.shape[1] == Theta.shape[0])
    h = X @ Theta

    try:
        assert (h.shape == Y.shape)
    except:
        print('h.shape {} !=Y.shape {}'.format(h.shape, Y.shape))

    J = 1 / 2 * np.sum(((h - Y) * R) ** 2) + lambd / 2 * np.sum(X ** 2) + lambd / 2 * np.sum(Theta ** 2)

    try:
        assert (len(J.shape) == 0)
    except:
        print('J is not raw number. J.shape = ', J.shape)

    return J

def J_derivative(Y, R, X, Theta, num_movies, num_users, num_features, lambd):
    cost_matr = (X @ Theta - Y) * R  # n_movies * n_users

    X_grad = cost_matr @ Theta.T
    Theta_grad = (cost_matr.T @ X).T

    try:
        assert (X_grad.shape == X.shape)
        assert (Theta_grad.shape == Theta.shape)
    except:
        print('Check gradient calculus')

    # Regularization part :
    X_grad += lambd * X
    Theta_grad += lambd * Theta

    return X_grad, Theta_grad

def fit(Y, R, num_features=10, alpha=0.0001, lambd=.01, eps=.1, max_iter=1000, step=100, verbose=0):
    num_movies, num_users = Y.shape

    if verbose:
        print('Running gradient descent with alpha= {}, lambda= {}, eps= {}, max_iter= {}'.format(
            alpha, lambd, eps, max_iter))

    #     X= params[:num_movies*num_features].reshape(num_movies,num_features)
    #     Theta = params[num_movies*num_features:].reshape(num_features,num_users)

    np.random.seed(2019)
    X = np.random.randn(num_movies, num_features)
    Theta = np.random.randn(num_features, num_users)

    J_hist = [-1]  # used for keeping J values. Init with -1 to avoid 0 at first iter
    continue_iter = True  # flag to continue next iter (grad desc step)
    iter_number = 0  # used for limit by max_iter

    while continue_iter:
        # Do step of gradient descent
        X_grad, Theta_grad = J_derivative(Y, R, X, Theta, num_movies, num_users, num_features, lambd)
        X = X - alpha * X_grad
        Theta = Theta - alpha * Theta_grad

        # keep history of J values
        J_hist.append(J(Y, R, X, Theta, lambd))
        # check criteria of exit (finish grad desc)
        if iter_number > max_iter:  # if limit succeeded
            continue_iter = False
            print('iter_number> max_iter')
        elif np.abs(J_hist[iter_number - 1] - J_hist[iter_number]) < eps:  # if accuracy is succeeded
            continue_iter = False
            print('J_hist[iter_number]={}'.format(J_hist[iter_number]))
        iter_number += 1

        if verbose and iter_number % step == 0:
            print('{}: {}'.format(iter_number, J_hist[iter_number - 1]))

    return X, Theta, J_hist


def Get_New_Data():
    array = np.random.randint(5, size=(15, 1))
    print(array)
    #data = pd.DataFrame(array)
    return array

def draw_cost_changes(J_hist):
    J_hist=J_hist[1:]
    plt.figure()
    plt.scatter(np.arange(0,len(J_hist)),J_hist,s=20,marker='.',c='b')
    plt.xlabel('Iterations')
    plt.ylabel('Cost')
    plt.show()


def add_my_ratings(Y, R, my_ratings):
    Y = np.c_[my_ratings, Y]
    R = np.c_[my_ratings != 0, R]
    return Y, R


df = GetData()
print(df)

R = np.vectorize(lambda x: 1 if x>.5 else 0)(df)
Y = df

#print(R)
#print(Y_scaled)

my_ratings = Get_New_Data()
print("my_ratings:",my_ratings)

Y, R = add_my_ratings(Y,R,my_ratings)

X, Theta, J_hist = fit(Y, R, num_features=20, alpha=0.0005, lambd=1, max_iter=1000,
                           eps=.01, step=50, verbose=1)

draw_cost_changes(J_hist)

pred= X@ Theta
my_pred = pred[:,0]

for i in range(15):
    print('Predicting rating {:.2} , provided: {}'.format(my_pred[i], my_ratings[i,0]))