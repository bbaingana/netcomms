import numpy as np
import scipy as sp
import networkx as nx
import matplotlib.pyplot as plt 

from static.statcoms import CommunityDetector

G = nx.karate_club_graph()

cd = CommunityDetector(G,2)

print  cd.get_hard_communities()

U = cd.get_soft_communities()

print U

clist = ["#FF6600", "#FFFF00"]
cd.comm_stack_plot(clist)



