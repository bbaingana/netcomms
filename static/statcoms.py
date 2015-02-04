import numpy as np
import scipy as sp
import math
import networkx as nx


class CommunityDetector:
    """This class leverages non-linear matrix factorization
       to unveil underlying communities in a given network"""
    def __init__(self, G, K=10):
	self.G = G
        self.A = nx.adj_matrix(G)
        self.ncomms = K
	self.community_affiliations =np.asmatrix(np.zeros((len(self.G.nodes()),self.ncomms)))
    
    def __nmf_comms(self):
        try:
            from sklearn.decomposition import ProjectedGradientNMF
	except ImportError:
	    print("Module sklearn is a required depenency.")
	    return
        model = ProjectedGradientNMF(n_components=self.ncomms, init='nndsvd')
	U = model.fit_transform(self.A)
	V = model.components_
	U = np.asmatrix(U)
	V = np.asmatrix(V)
	V = V.T
	self.community_affiliations = U

    def get_soft_communities(self):
        self.__nmf_comms()
	return self.community_affiliations
    
    def __comm_map(self):
	community_map = {}
	node_list = self.G.nodes()
	for i in range(self.community_affiliations.shape[0]):
            community_map[node_list[i]] = self.community_affiliations[i,:].argmax()
	return community_map

    def get_hard_communities(self):
        self.__nmf_comms()
        comm_dict = self.__comm_map()
	return comm_dict

    
    def comm_stack_plot(self, color_list):
	try:
	    import matplotlib.pyplot as plt
	except ImportError:
	    print("Matplotlib module is needed to plot graphs.")
	    return

        assert len(color_list) == self.ncomms, \
	       "Insufficient number of colors\n"

	X = self.community_affiliations
	for i in range(X.shape[0]):
	    X[i,:] = X[i,:]/X[i,:].sum()
	
	data = np.array(X.T)
	bottom = np.cumsum(data, axis=0)
	node_list = range(X.shape[0])

	plt.bar(node_list, data[0], color=color_list[0])

	for j in range(1, X.shape[1]):
	    plt.bar(node_list, data[j], bottom=bottom[j-1],\
		    color=color_list[j])
	plt.ylim((0,1))
	plt.show()


class RobustCommunityDetector:
    
    def __init__(self):
        pass

