import numpy as np
import scipy as sp
import math


class CommunityTracker:
    
    def __init__(self):
        pass


class RobustCommunityTracker:
    
    def __init__(self):
        pass

    def __soft_thresh(M, mu):
        P = np.asmatrix(np.zeros(M.shape))
	for i in range(M.shape[0]):
            for j in range(M.shape[1]):
                P[i,j] = np.sign(M[i,j])*max(abs(M[i,j]) - mu, 0)
	
	return P

    def __proj_onto_nnorthant(M):
        """Simple function to project numpy matrix onto non-negative orthant"""
	P = np.asmatrix(np.zeros((M.shape[0], M.shape[1])))
	for i in range(M.shape[0]):
	    for j in range(M.shape[1]):
	        if M[i,j] > 0:
		    P[i,j] = M[i,j]
	return P

    def __max_eigenvalue(M):
        eigvals, eigvecs = np.linalg.eig(M)
	return max(eigvals)

    def __nmf_initialization(A, ncomms):
        try:
	    from sklearn.decomposition import ProjectedGradientNMF
	except ImportError:
	    print("sklearn module is missing.")
	    return
        
        model = ProjectedGradientNMF(n_components=ncomms, init='nndsvd')
	Uin = np.asmatrix(model.fit_transform(A))
	Vin = np.asmatrix(model.components_)
	Vin = Vin.T
	init_dict = {'U':Uin, 'V':Vin}
	return init_dict

    def __svd_initialization(A, ncomms):
        from scipy import linalg
	U, s, Vt = linalg.svd(A)
	U = np.asmatrix(U)
	Vt = np.asmatrix(Vt)
	S = np.asmatrix(linalg.diagsvd(s, A.shape[0], A.shape[1]))
	Ss = S[:, 0:ncomms]
	Vts = Vt[0:ncomms, :]
	F1 = U*Ss
	init_dict = {'U':F1, 'V':Vts.T}
	return init_dict

