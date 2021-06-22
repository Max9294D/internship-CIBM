import numpy.random as random
import numpy as np
from scipy import special
import matplotlib.pyplot as plt
import pandas as pd
import csv

def Wpar(ak,ad,md):
    # """ les arguments doivent être des arrays"""
    # if type(ak) != np.ndarray or type(ad) != np.ndarray or type(md) != np.ndarray:
    #     return "Error -- the arguments are not arrays"

    return np.array((ak * ((ad/md)**2)))

def Wpar_2(W0,W2,W4):
    # """ les arguments doivent être des arrays"""
    # if type(W0) != np.ndarray or type(W2) != np.ndarray or type(W4) != np.ndarray:
    #     return "Error -- the arguments are not arrays"

    return np.array((W0+W2+W4))

def Wperp(rk,rd,md):
    # """ les arguments doivent être des arrays"""
    # if type(rk) != np.ndarray or type(rd) != np.ndarray or type(md) != np.ndarray:
    #     return "Error -- the arguments are not arrays"

    return np.array((rk * (rd/md)**2))

def Wperp_2(W0,W2,W4):
    # """ les arguments doivent être des arrays"""
    # if type(W0) != np.ndarray or type(W2) != np.ndarray or type(W4) != np.ndarray:
    #     return "Error -- the arguments are not arrays"

    return np.array((W0 - W2/2 + 3*W4/8))

def D0(md):
    """
    the Legendre expansion coefficient of the diffusion tensor D0
    # les arguments doivent être des arrays
    """
    # if type(md) != np.ndarray:
    #     return "Error -- the arguments are not arrays"


    return np.array((md))

def D0_2(f,Dapar,Depar,Deper):
    """
    the Legendre expansion coefficient of the diffusion tensor D0
    les arguments doivent être des arrays
    """
    # if type(f) != np.ndarray or type(Dapar) != np.ndarray or type(Depar) != np.ndarray or type(Deper) != np.ndarray:
    #     return "Error -- the arguments are not arrays"

    return np.array((1/3 * (f*Dapar + (1-f) * (Depar + 2*Deper))))

def D2(ad,rd):
    """
    the Legendre expansion coefficient of the diffusion tensor D2
    # les arguments doivent être des arrays
    """
    # if type(ad) != np.ndarray or type(rd) != np.ndarray:
    #     return "Error -- the arguments are not arrays"

    return np.array((2/3 * (ad - rd)))

def D2_2(f,Dapar,Depar,Deper,kappa):
    """
    the Legendre expansion coefficient of the diffusion tensor D2
    les arguments doivent être des arrays
    """
    # if type(f) != np.ndarray or type(Dapar) != np.ndarray or type(Depar) != np.ndarray or type(Deper) != np.ndarray or type(kappa) != np.ndarray:
    #     return "Error -- the arguments are not arrays"

    return np.array((2/3 * p2(kappa) * (f*Dapar + (1-f) * (Depar - Deper))))

def W0(mk):
    """
    the Legendre expansion coefficient of the kurtosis tensor W0
    # les arguments doivent être des arrays
    """
    # if type(mk) != np.ndarray:
    #     return "Error -- the arguments are not arrays"


    return np.array((mk))

def W0_2(f,Dapar,Depar,Deper):
    """
    the Legendre expansion coefficient of the kurtosis tensor W0
    les arguments doivent être des arrays
    """
    # if type(f) != np.ndarray or type(Dapar) != np.ndarray or type(Depar) != np.ndarray or type(Deper) != np.ndarray:
    #     return "Error -- the arguments are not arrays"

    A = 5*Deper**2 + (Depar-Deper)**2 +10/3 * Deper * (Depar - Deper)
    B = f * Dapar**2 + (1-f)*A - D0_2(f,Dapar,Depar,Deper)**2
    C = 1/(5*D0_2(f,Dapar,Depar,Deper)**2) * B - 1
    D = 3 * C
    return np.array((D))

def W2(ak,ad,md,mk,rk,rd):
    """
    the Legendre expansion coefficient of the kurtosis tensor W2
    # les arguments doivent être des arrays
    """
    # if type(ak) != np.ndarray or type(ad) != np.ndarray or type(md) != np.ndarray or type(mk) != np.ndarray or type(rk) != np.ndarray or type(rd) != np.ndarray:
    #     return "Error -- the arguments are not arrays"

    return np.array((1/7 * (3*Wpar(ak,ad,md) + 5*mk - 8*Wperp(rk,rd,md))))

def W2_2(f,Dapar,Depar,Deper,kappa):
    """
    the Legendre expansion coefficient of the kurtosis tensor W2
    les arguments doivent être des arrays
    """
    # if type(f) != np.ndarray or type(Dapar) != np.ndarray or type(Depar) != np.ndarray or type(Deper) != np.ndarray or type(kappa) != np.ndarray:
    #     return "Error -- the arguments are not arrays"

    A = (Depar - Deper)**2 + 7/3 * Deper * (Depar - Deper)
    B = f * Dapar**2 + (1-f)*A
    C = p2(kappa) * B -1/2 * D2_2(f,Dapar,Depar,Deper,kappa) * (D2_2(f,Dapar,Depar,Deper,kappa) + 7*D0_2(f,Dapar,Depar,Deper))
    D = 12/(7*D0_2(f,Dapar,Depar,Deper)**2) * C
    return np.array((D))

def W4(ak,ad,md,mk,rk,rd):
    """
    the Legendre expansion coefficient of the kurtosis tensor W4
    # les arguments doivent être des arrays
    """
    # if type(ak) != np.ndarray or type(ad) != np.ndarray or type(md) != np.ndarray or type(mk) != np.ndarray or type(rk) != np.ndarray or type(rd) != np.ndarray:
    #     return "Error -- the arguments are not arrays"

    return np.array((4/7 * (Wpar(ak,ad,md) - 3*mk + 2*Wperp(rk,rd,md))))

def W4_2(f,Dapar,Depar,Deper,kappa):
    """
    the Legendre expansion coefficient of the kurtosis tensor W4
    les arguments doivent être des arrays
    """
    # if type(f) != np.ndarray or type(Dapar) != np.ndarray or type(Depar) != np.ndarray or type(Deper) != np.ndarray or type(kappa) != np.ndarray:
    #     return "Error -- the arguments are not arrays"

    A = f * Dapar**2 + (1-f)*(Depar - Deper)**2
    B = p4(kappa)*A - 9/4 * D2_2(f,Dapar,Depar,Deper,kappa)**2
    C = B * 24 / (35*D0_2(f,Dapar,Depar,Deper)**2)
    return np.array((C))

def F(x):
    """
    Watson distribution
    # les arguments doivent être des arrays
    """
    # if type(x) != np.ndarray:
    #     return "Error -- the arguments are not arrays"

    return np.array((np.sqrt(np.pi) * np.exp(-x**2) * special.erfi(x) / 2))

def c2(kappa):
    """
    Watson distribution
    les arguments doivent être des arrays
    """
    # if type(kappa) != np.ndarray:
    #     return "Error -- the arguments are not arrays"

    return np.array((1 / (2 * np.sqrt(kappa) * F(np.sqrt(kappa))) - 1 / (2*kappa)))

def p2(kappa):
    """
    2nd order spherical harmonics expansion coefficients of the axially-symetric ODF
    les arguments doivent être des arrays
    """
    # if type(kappa) != np.ndarray:
    #     return "Error -- the arguments are not arrays"


    return np.array(((3*c2(kappa) - 1) / 2))

def p4(kappa):
    """
    4th order spherical harmonics expansion coefficients of the axially-symetric ODF
    les arguments doivent être des arrays
    """
    # if type(kappa) != np.ndarray:
    #     return "Error -- the arguments are not arrays"


    return np.array((c2(kappa) * (5/8 - 105/(16*kappa)) + 35/(16*kappa) + 3/8))

def AD(D0,D2):
    """
    axial diffusivity
    # les arguments doivent être des arrays
    """
    # if type(D0) != np.ndarray or type(D2) != np.ndarray:
    #     return "Error -- the arguments are not arrays"


    return np.array((D0+D2))

def RD(D0,D2):
    """
    radial diffusivity
    # les arguments doivent être des arrays
    """
    # if type(D0) != np.ndarray or type(D2) != np.ndarray:
    #     return "Error -- the arguments are not arrays"


    return np.array((D0-D2/2))

def MD(D0):
    return np.array((D0))

def AK(D0,D2,W0,W2,W4):
    """
    axial excess kurtosis
    # les arguments doivent être des arrays
    """
    # if type(D0) != np.ndarray or type(D2) != np.ndarray or type(W0) != np.ndarray or type(W2) != np.ndarray or type(W4) != np.ndarray:
    #     return "Error -- the arguments are not arrays"


    return np.array(((D0/AD(D0,D2))**2 * Wpar_2(W0,W2,W4)))

def RK(D0,D2,W0,W2,W4):
    """
    radial excess kurtosis
    # les arguments doivent être des arrays
    """
    # if type(D0) != np.ndarray or type(D2) != np.ndarray or type(W0) != np.ndarray or type(W2) != np.ndarray or type(W4) != np.ndarray:
    #     return "Error -- the arguments are not arrays"


    return np.array((((D0/RD(D0,D2))**2)*Wperp_2(W0,W2,W4)))

def MK(W0):
    return np.array((W0))

# def table_correspondance_c2_kappa(c_2):
#     kappa = np.arange(2,65,0.01)
#     C2 = []
#     KAPPA = []
#     for i in kappa:
#         C2.append(c2(i))
#         KAPPA.append(i)
#     # C2 = np.asarray(C2)
#     diff_c2 = 10000
#     kappa_retenu = 100000
#     for i in range(len(C2)):
#         if abs(C2[i] - c_2) < diff_c2:
#             diff_c2 = abs(C2[i] - c_2)
#             kappa_retenu = KAPPA[i]
#     return kappa_retenu

def create_tableau_correspondance_c2_kappa():
    "créer un fichier csv de correspondance entre les valeurs de c2 et de kappa"

    kappa = np.arange(2,128,0.01)
    C2 = []
    KAPPA = []
    for i in kappa:
        C2.append(c2(i))
        KAPPA.append(i)
    dict_correspondance = {'c2': C2, 'kappa': KAPPA}

    df = pd.DataFrame(dict_correspondance)
    df.to_csv('tab_correspondance.csv')
    return True

def interrogate_tableau_correspondance_c2_kappa(filename, c_2):
    "renvoit le kappa associé au c2 en argument en interrogeant le fichier csv passé en argument"

    with open(filename, newline='') as f:  # Ouverture du fichier CSV
        C2 = []
        KAPPA = []
        lire = csv.reader(f)  # chargement des lignes du fichier csv
        for ligne in lire:  # Pour chaque ligne...
            if ligne != ['','c2','kappa']: #on ne prend pas en compte la première ligne
                C2.append(float(ligne[1]))
                KAPPA.append(float(ligne[2]))

    diff_c2 = 10000
    kappa_retenu = 100000
    for i in range(len(C2)):
        if abs(C2[i] - c_2) < diff_c2:
            diff_c2 = abs(C2[i] - c_2)
            kappa_retenu = KAPPA[i]
    return kappa_retenu

if __name__ == "__main__":



    # à décommenter si on n'a pas encore créer le tableau de correspondance puis recommenter juste après
    # create_tableau_correspondance_c2_kappa()

    #test correspondance tableau
    filename = 'tab_correspondance.csv'
    c_2 = random.uniform(1/3,1)
    kappa = interrogate_tableau_correspondance_c2_kappa(filename, c_2)
    print("c2 : ",c_2," kappa : ", kappa)

