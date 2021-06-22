import numpy as np
import numpy.random as random
import fct_equations2
from scipy.io import savemat

def generate_parameters(nb_iter):
    """
    :param nb_iter: entier positif
    :return: f, kappa, Da, Depar, Deperp des listes contenant les valeurs des paramètres du même nom générés aléatoirement
    """
    # on s'assure que les conditions sur nb_iter sont respectées
    if nb_iter<0:
        nb_iter = round(-nb_iter)
    else:
        nb_iter = round(nb_iter)

    parametres = np.zeros((nb_iter,5))

    for i in range(nb_iter):

        f = random.uniform(0,1)
        c2 =random.uniform(1/3,1)
        Da = random.uniform(1.5,3)
        Depar = random.uniform(0.5,2)
        Deperp = random.uniform(0,1.5)

        parametres[i,0] = f
        parametres[i,1] = Da
        parametres[i,2] = Depar
        parametres[i,3] = Deperp
        parametres[i,4] = c2

    return parametres


def generate_Legendre_coef(parametres):
    """
    :param f, kappa, Da, Depar, Deperp des listes contenant les valeurs des paramètres du même nom générés aléatoirement
    :return: D0, D2, W0, W2, W4 les coefficients de Legendre associés
    """

    param = np.zeros_like(parametres)
    filename = 'tab_correspondance.csv'
    for i in range(param.shape[0]):

        f = parametres[i,0]
        Da = parametres[i,1]
        Depar = parametres[i,2]
        Deperp = parametres[i,3]
        c2 = parametres[i,4]

        kappa = fct_equations2.interrogate_tableau_correspondance_c2_kappa(filename, c2)

        D0 = fct_equations2.D0_2(f,Da,Depar,Deperp)
        D2 = fct_equations2.D2_2(f, Da, Depar, Deperp,kappa)
        W0 = fct_equations2.W0_2(f, Da, Depar, Deperp)
        W2 = fct_equations2.W2_2(f, Da, Depar, Deperp,kappa)
        W4 = fct_equations2.W4_2(f, Da, Depar, Deperp,kappa)

        param[i,0] = D0
        param[i,1] = D2
        param[i,2] = W0
        param[i,3] = W2
        param[i,4] = W4

    return param

def generate_moment(param):
    """
    :param D0, D2, W0, W2, W4 les coefficients de Legendre
    :return: AD, RD, MD, AK, RK, Mk les tenseurs des moments associés
    """

    p = np.zeros((param.shape[0],param.shape[1]+1))

    for i in range(param.shape[0]):

        D0 = param[i, 0]
        D2 = param[i, 1]
        W0 = param[i, 2]
        W2 = param[i, 3]
        W4 = param[i, 4]

        AD = fct_equations2.AD(D0,D2)
        RD = fct_equations2.RD(D0,D2)
        MD = fct_equations2.MD(D0)
        AK = fct_equations2.AK(D0,D2,W0,W2,W4)
        RK = fct_equations2.RK(D0,D2,W0,W2,W4)
        MK = fct_equations2.MK(W0)

        p[i, 0] = MD
        p[i, 1] = AD
        p[i, 2] = RD
        p[i, 3] = MK
        p[i, 4] = AK
        p[i, 5] = RK


    return p


def normalize(x):
    """
    Transform an tensor (shape,) into a tensor of shape (shape, 1)

    Arguments
    Tensor.

    Returns:
    result -- Transformed tensor
    """

    norme = np.linalg.norm(x, axis=0)
    x = x/norme

    return norme, x


def train_set(nb_elements): #equivalente à dev_set et test_set mais c'est pour augmenter la compréhension du code
    """
    :param nb_elements: entier positif
    :return: x_train et y_train deux datasets prêts à l'emploi chacun composé de nb_elements éléments
    """
    parameters = generate_parameters(nb_elements)
    param = generate_Legendre_coef(parameters)
    p = generate_moment(param)
    x_train = p
    y_train = parameters
    return x_train, y_train

def dev_set(nb_elements):
    """
    :param nb_elements: entier positif
    :return: x_dev et y_dev deux datasets prêts à l'emploi chacun composé de nb_elements éléments
    """
    parameters = generate_parameters(nb_elements)
    param = generate_Legendre_coef(parameters)
    p = generate_moment(param)
    x_dev = p
    y_dev = parameters
    return x_dev, y_dev

def test_set(nb_elements):
    """
    :param nb_elements: entier positif
    :return: x_test et y_test deux datasets prêts à l'emploi chacun composé de nb_elements éléments
    """
    parameters = generate_parameters(nb_elements)
    param = generate_Legendre_coef(parameters)
    p = generate_moment(param)
    x_test = p
    y_test = parameters
    return x_test, y_test

def generate_sets(nb_elements, prop_train, prop_dev, prop_test):
    """
    :param nb_elements: entier positif représentant le nombre total d'éléments pour le train set, le dev set et le test set
    :param prop_train: proportion d'éléments réservés pour le train set
    :param prop_dev: proportion d'éléments réservés pour le dev set
    :param prop_test: proportion d'éléments réservés pour le test set
    :return:  x_train, y_train, x_dev, y_dev, x_test et y_test six datasets prêts à l'emploi chacun composé de la proportion de nb_elements éléments
    """
    x_train, y_train = train_set(nb_elements * prop_train)
    x_dev, y_dev = dev_set(nb_elements * prop_dev)
    x_test, y_test = test_set(nb_elements * prop_test)
    # norme_x_train, new_x_train = normalize(x_train)
    # norme_y_train, new_y_train = normalize(y_train)
    # norme_x_dev, new_x_dev = normalize(x_dev)
    # norme_y_dev, new_y_dev = normalize(y_dev)
    # norme_x_test, new_x_test = normalize(x_test)
    # norme_y_test, new_y_test = normalize(y_test)
    # return new_x_train, new_y_train, new_x_dev, new_y_dev, new_x_test, new_y_test, norme_x_train, norme_y_train, norme_x_dev, norme_y_dev, norme_x_test, norme_y_test
    return x_train, y_train, x_dev, y_dev, x_test, y_test

def conversion_matlab(nb_elements, prop_train):
    x_train, y_train, x_dev, y_dev, x_test, y_test = generate_sets(nb_elements, prop_train, 0, 1-prop_train)

    # traitement pour un enregistrement des données dans un format mat
    mat_dic = {'dki': x_train,
               'wmti_paras': y_train}

    savemat("datasets/wmti_dki_constraint2_test_artificiel_c2.mat", mat_dic)

    return True

if __name__ == "__main__":

    conversion_matlab(10000, 0.8)



