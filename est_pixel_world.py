import numpy as np

def est_pixel_world(pixels, R_wc, t_wc, K):
    """
    Estimate the world coordinates of a point given a set of pixel coordinates.
    The points are assumed to lie on the x-y plane in the world.
    Input:
        pixels: N x 2 coordiantes of pixels
        R_wc: (3, 3) Rotation of camera in world
        t_wc: (3, ) translation from world to camera
        K: 3 x 3 camara intrinsics
    Returns:
        Pw: N x 3 points, the world coordinates of pixels
    """

    ##### STUDENT CODE START #####
    print("pixels", pixels)
    print("R_wc", R_wc)
    R_wc_inv = np.linalg.inv(R_wc)
    print(R_wc_inv)
    r1 = (R_wc_inv[:,0])
    r2 = (R_wc_inv[:,1])
    t_wc = -(np.matmul(R_wc_inv,t_wc))
    print("t_wc= {}".format(t_wc))
    print("r1= {}".format(r1))
    print("r2= {}".format(r2))
    N = pixels.shape[0]
    # print(pixels)
    # pixels = [pixels,np.ones(N)]
    pixels_ = np.ones((N,3))
    print(pixels_)
    pixels_[:,0] = pixels[:,0]
    pixels_[:,1] = pixels[:,1]
    print("pixels",pixels_)
    # print(R_wc_inv.shape)
    # print(r1.shape)
    R1_2_twc = np.array([r1,r2,t_wc])
    print("R1_2_twc_transposed= ",np.transpose(R1_2_twc))
    print(np.linalg.inv(np.matmul(K,np.transpose(R1_2_twc))).shape)
    #
    # Pw = np.matmul(pixels_,np.linalg.inv(np.matmul(K,np.transpose(R1_2_twc))))
    # Pw = Pw / Pw[:, -1][:, None]
    # Pw[:, -1] = 0
    Pw = np.zeros((pixels_.shape[0],3))

    for i in range(pixels.shape[0]):
        Pw[i,:]=np.transpose(np.linalg.inv(K@R1_2_twc.T)@pixels_[i,:].T)
        Pw[i,:]=Pw[i,:]/Pw[i,-1]
        Pw[i,-1]=0
    #Pw = np.matmul(pixels_,np.linalg.inv(np.matmul(K,R1_2_twc)))
    ##### STUDENT CODE END #####
    return Pw
