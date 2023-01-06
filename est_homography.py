import numpy as np

def est_homography(X, Y):
    """
    Calculates the homography H of two planes such that Y ~ H*X
    If you want to use this function for hw5, you need to figure out
    what X and Y should be.
    Input:
        X: 4x2 matrix of (x,y) coordinates
        Y: 4x2 matrix of (x,y) coordinates
    Returns:
        H: 3x3 homogeneours transformation matrix s.t. Y ~ H*X

    """

    ##### STUDENT CODE START #####
    # X = np.array(X)
    # Y = np.array(Y)
    # print(X.shape)
    # print(Y.shape)
    x = X[:, 0]
    y = X[:, 1]
    xP = Y[:, 0]
    yP = Y[:, 1]
    #ax=np.zeros([4,9])
    #ay=np.zeros([4,9])
    A = []
    for i in range(X.shape[0]):
        A.append([-x[i], -y[i], -1, 0, 0, 0, x[i]*xP[i], y[i]*xP[i], xP[i]])
        A.append([0, 0, 0, -x[i], -y[i], -1, x[i]*yP[i], y[i]*yP[i], yP[i]])
    A = np.array(A)
    [U, S, Vt] = np.linalg.svd(A)
    h = np.transpose(Vt)[:,-1]
    H = h.reshape(3,3)
    ##### STUDENT CODE END #####

    return H
