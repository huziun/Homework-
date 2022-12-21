import matplotlib.pyplot as plt;
def plot_data_logistic_regression(X, y, legend_loc=1, title=None):
    '''
    :param X: 2 dimensional ndarray
    :param y:  1 dimensional ndarray. Use y.ravel() if necessary
    :return:
    '''

    positive_indices = (y == 1)
    negative_indices = (y == 0)
    #     import matplotlib as mpl
    colors_for_points = ['grey', 'orange']  # neg/pos

    plt.scatter(X[negative_indices][:, 0], X[negative_indices][:, 1], s=40, c=colors_for_points[0], edgecolor='black',
                label='negative', alpha=0.7)
    plt.scatter(X[positive_indices][:, 0], X[positive_indices][:, 1], s=40, c=colors_for_points[1], edgecolor='black',
                label='positive', alpha=0.7)
    plt.title(title)
    plt.legend(loc=legend_loc)

