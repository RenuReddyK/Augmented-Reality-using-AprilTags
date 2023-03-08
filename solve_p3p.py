import numpy as np

def P3P(Pc, Pw, K=np.eye(3)):
    """
    Solve Perspective-3-Point problem, given correspondence and intrinsic

    Input:
        Pc: 4x2 numpy array of pixel coordinate of the April tag corners in (x,y) format
        Pw: 4x3 numpy array of world coordinate of the April tag corners in (x,y,z) format
    Returns:
        R: 3x3 numpy array describing camera orientation in the world (R_wc)
        t: (3,) numpy array describing camera translation in the world (t_wc)

    """

    # Invoke Procrustes function to find R, t
    # Need to select the R and t that could transoform all 4 points correctly.
    # R,t = Procrustes(Pc_3d, Pw[1:4])

    u = Pc[:,0]
    v = Pc[:,1]
    
    Pc_ = np.ones((3,3))
    Pc_[:,0] = Pc[1:4,0]
    Pc_[:,1] = Pc[1:4,1]

    P = np.matmul(np.linalg.inv(K),np.transpose(Pc_))

    j1 = P[:,0]/np.linalg.norm(P[:,0])
    j2 = P[:,1]/np.linalg.norm(P[:,1])
    j3 = P[:,2]/np.linalg.norm(P[:,2])

    cos_alpha = np.dot(j2,j3)
    cos_beta = np.dot(j1,j3)
    cos_gama = np.dot(j1,j2)
    print(cos_alpha)
    print(cos_beta)
    print(cos_gama)
    P1 = Pw[1,:]
    P2 = Pw[2,:]
    P3 = Pw[3,:]
    a = np.linalg.norm(P2-P3)
    b = np.linalg.norm(P1-P3)
    c = np.linalg.norm(P1-P2)
    a2 = a*a
    b2 = b*b
    c2 = c*c
    cos_alpha2 = cos_alpha*cos_alpha
    cos_beta2 = cos_beta* cos_beta
    cos_gama2 = cos_gama*cos_gama
    A4 = ((a2- c2)/(b2) - 1)**2 - (4*c2*cos_alpha2)/(b2)
    A3 = 4*(((a2-c2)/b2)*(1-((a2-c2)/b2))*cos_beta -(1-((a2+c2)/b2))*cos_alpha*cos_gama + 2*c2*cos_alpha2*cos_beta/b2)
    A2 = 2*((((a2-c2)/b2)**2) -1 + 2*(((a2-c2)/b2)**2)*cos_beta2 + 2*(((b2-c2)/b2))*cos_alpha2 -4*(((a2+c2)/b2)*cos_alpha*cos_beta*cos_gama) + 2*((b2*cos_gama2 -a2*cos_gama2)/b2))
    A1 = 4*((-(a2-c2)/b2)*(1 + ((a2-c2)/b2))*cos_beta + 2*(a2/b2)*cos_gama2*cos_beta-(1-((a2+c2)/b2))*cos_alpha *cos_gama)
    A0 = (1 + (a2-c2)/b2)**2 - (4*a2*cos_gama2)/b2
    A = [A4, A3, A2, A1, A0]
    print("A",A)
    roots = np.roots(A)
    print("Roots",roots)

    roots = roots[np.isreal(roots)]
    roots = roots[roots > 0]
    roots = np.real(roots)
    print(roots)

    R_ = None
    t_ = None
    minerr = float('inf')
    for i in range(len(roots)):
        u = ((-1 +(a2-c2)/b2)*np.square(roots[i]) - 2*((a2-c2)/b2)*cos_beta*roots[i] +1+((a2-c2)/b2))/(2*(cos_gama-roots[i]*cos_alpha))

        print("u",u)
        s1 = np.sqrt((c2)/(1+u**2 -2*u*cos_gama))
        s2 = u * s1
        s3 = roots[i] * s1
        p = np.vstack((s1*j1, s2*j2, s3*j3))
        R, t = Procrustes(Pw[1:,:], p)

        P = np.matmul(K, np.matmul(R, np.transpose(Pw[0,:])) + t)
        P = (P/P[-1])[:-1]
        error = np.linalg.norm(P-Pc[0, :])

        if error < minerr:
            minerr = error
            R_ = R
            t_ = t

    return np.linalg.inv(R_),(-np.linalg.inv(R_)@t_)

    #return R, t

def Procrustes(X, Y):
    """
    Solve Procrustes: Y = RX + t

    Input:
        X: Nx3 numpy array of N points in camera coordinate (returned by your P3P)
        Y: Nx3 numpy array of N points in world coordinate
    Returns:
        R: 3x3 numpy array describing camera orientation in the world (R_wc)
        t: (3,) numpy array describing camera translation in the world (t_wc)

    """
    B = X
    A = Y
    A_mean = np.mean(A,axis =0)
    B_mean = np.mean(B,axis =0)
    A = np.transpose(A - A_mean)
    B = np.transpose(B - B_mean)

    R = np.matmul(A,np.transpose(B))
    [U, S, Vt] = np.linalg.svd(R)
    R = np.matmul(np.matmul(U,[[1,0,0],[0,1,0],[0,0,np.linalg.det(np.matmul(np.transpose(Vt),np.transpose(U)))]]),Vt)
    t = A_mean - np.matmul(R,B_mean)

    return R, t
