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

    ##### STUDENT CODE START #####

    # Invoke Procrustes function to find R, t
    # # You may need to select the R and t that could transoform all 4 points correctly.
    # R,t = Procrustes(Pc_3d, Pw[1:4])

    u = Pc[:,0]
    v = Pc[:,1]
    print("u",u)
    print("v",v)
    # f = (K[0,0]+K[1,1])/2
    # u1 = u[0]
    # u2 = u[1]
    # u3 = u[2]
    # v1 = v[0]
    # v2 = v[1]
    # v3 = v[2]
    # u1_square = np.matmul(u1,np.transpose(u1))
    # v1_square = np.matmul(v1,np.transpose(v1))
    # j1 = (1/(np.sqrt(np.square(np.matmul(u1,np.transpose(u1)))+np.square(np.matmul(v1,np.transpose(v1)))+f*f)))*[u1,v1,f]
    # j2 = (1/(np.sqrt(np.square(np.matmul(u2,np.transpose(u2)))+np.square(np.matmul(v2,np.transpose(v2)))+f*f)))*[u2,v2,f]
    # j3 = (1/(np.sqrt(np.square(np.matmul(u3,np.transpose(u3)))+np.square(np.matmul(v3,np.transpose(v3)))+f*f)))*[u3,v3,f]

    Pc_ = np.ones((3,3))
    Pc_[:,0] = Pc[1:4,0]
    Pc_[:,1] = Pc[1:4,1]
    # print(np.linalg.inv(K).shape)
    # print(Pc_.shape)
    P = np.matmul(np.linalg.inv(K),np.transpose(Pc_))
    print(P)
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
    print(a)
    print(b)
    print(c)
    a2 = a*a
    b2 = b*b
    c2 = c*c
    cos_alpha2 = cos_alpha*cos_alpha
    cos_beta2 = cos_beta* cos_beta
    cos_gama2 = cos_gama*cos_gama
    A4 = ((a2- c2)/(b2) - 1)**2 - (4*c2*cos_alpha2)/(b2)
    print("A4",A4)
    A3 = 4*(((a2-c2)/b2)*(1-((a2-c2)/b2))*cos_beta -(1-((a2+c2)/b2))*cos_alpha*cos_gama + 2*c2*cos_alpha2*cos_beta/b2)
    print("A3",A3)
    A2 = 2*((((a2-c2)/b2)**2) -1 + 2*(((a2-c2)/b2)**2)*cos_beta2 + 2*(((b2-c2)/b2))*cos_alpha2 -4*(((a2+c2)/b2)*cos_alpha*cos_beta*cos_gama) + 2*((b2*cos_gama2 -a2*cos_gama2)/b2))
    print("A2",A2)
    A1 = 4*((-(a2-c2)/b2)*(1 + ((a2-c2)/b2))*cos_beta + 2*(a2/b2)*cos_gama2*cos_beta-(1-((a2+c2)/b2))*cos_alpha *cos_gama)
    print("A1",A1)
    A0 = (1 + (a2-c2)/b2)**2 - (4*a2*cos_gama2)/b2
    print("A0",A0)
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
        print("s1",s1)
        s2 = u * s1
        print("s2",s2)
        s3 = roots[i] * s1
        print("s3",s3)
        p = np.vstack((s1*j1, s2*j2, s3*j3))
        R, t = Procrustes(Pw[1:,:], p)

        P = np.matmul(K, np.matmul(R, np.transpose(Pw[0,:])) + t)
        # P = K@(R@Pw[0, :].T + t)
        P = (P/P[-1])[:-1]
        error = np.linalg.norm(P-Pc[0, :])

        if error < minerr:
            minerr = error
            R_ = R
            t_ = t

    return np.linalg.inv(R_),(-np.linalg.inv(R_)@t_)

    ##### STUDENT CODE END #####

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
    ##### STUDENT CODE START #####
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
    print(R)
    return R, t
