import numpy as np 
from function_app1 import *

file = 'data_IG1.txt'
tab = np.genfromtxt(file, delimiter = ',', skip_header = 4)
cov, row = np. shape(tab)

if __name__ == "_main":
    # utworzenie obiektu
    geo = Transformacje(model = "wgs84")
    # dane XYZ geocentryczne
    X = []
    Y = []
    Z = []
    for i in range(0, len(tab)):
        XX = tab[i][0]
        YY = tab[i][1]
        ZZ = tab[i][2]
        X.append(XX)
        Y.append(YY)
        Z.append(ZZ)
        
        
    
   