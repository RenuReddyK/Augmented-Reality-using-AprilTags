from est_homography import est_homography
import numpy as np

def PnP(Pc, Pw, K=np.eye(3)):
    """
    Solve Perspective-N-Point problem with collineation assumption, given correspondence and intrinsic

    Input:
        Pc: 4x2 numpy array of pixel coordinate of the April tag corners in (x,y) format
        Pw: 4x3 numpy array of world coordinate of the April tag corners in (x,y,z) format
    Returns:
        R: 3x3 numpy array describing camera orientation in the world (R_wc)
        t: (3, ) numpy array describing camera translation in the world (t_wc)

    """

    ##### STUDENT CODE START #####

    # Homography Approach
    # Following slides: Pose from Projective Transformation
    H= est_homography(Pw,Pc)

    normalized_H = H/H[2,2]
    K_Inverse_normalized_H = np.matmul(np.linalg.inv(K),normalized_H)
    # print(normalized_K)
    h1 = np.transpose(K_Inverse_normalized_H[:3,0])
    h2 = np.transpose(K_Inverse_normalized_H[:3,1])
    h3 = np.cross(h1,h2)

    [U, S, Vt] = np.linalg.svd([h1,h2,h3])
    # norm = np.linalg.norm(np.array(K[:3,3]))
    # normalized_K = K/norm
    R=np.matmul(np.matmul(U,[[1,0,0],[0,1,0],[0,0,np.linalg.det(np.matmul(U,Vt))]]),Vt)
    print(R)

    #t = np.transpose(h3)/np.linalg.norm(np.transpose(h1))
    t = -np.matmul(R,(K_Inverse_normalized_H[:3,2]/(np.linalg.norm(h1))))
    print(t)

    ##### STUDENT CODE END #####

    return R, t
