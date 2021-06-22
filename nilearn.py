import math
import matplotlib.pyplot as plt
import numpy as np
import os
import nibabel as nib
from scipy.io import savemat, loadmat

def process_mat_matrix(path_mat_matrix):
    matrix = loadmat(path_mat_matrix)
    # print("output_result :",output_result)
    # print("type output_result :",type(output_result))
    output_result = matrix['output_final']
    measures_result = matrix['measure_final']
    print("output_result :",output_result)
    print("type output_result :",type(output_result))
    print("shape : ",output_result.shape)

    print("measures_result :",measures_result)
    print("type measures_result :",type(measures_result))
    print("shape : ",measures_result.shape)

    np.savetxt("output_result.csv", output_result, delimiter=",")
    np.savetxt("measure_result.csv", measures_result, delimiter=",")

    print(len(output_result))
    print(output_result.shape[0])

    i=0
    while i < output_result.shape[0]:  # on supprimer les zeros et les 'inf' en trop + filtrage des valeurs de Da trop faibles
        if (output_result[i,0] == 0 and output_result[i,1] == 0 and output_result[i,2] == 0 and output_result[i,3] == 0 and output_result[i,4] == 0) \
                or (math.isnan(output_result[i, 0]) or math.isnan(output_result[i, 1]) or math.isnan(output_result[i, 2]) or math.isnan(output_result[i, 3]) or math.isnan(output_result[i, 4])) \
                or (math.isnan(measures_result[i, 0]) or math.isnan(measures_result[i, 1]) or math.isnan( measures_result[i, 2]) or math.isnan(measures_result[i, 3]) or math.isnan( measures_result[i, 4]) or math.isnan(measures_result[i, 5])) \
                or (math.isinf(output_result[i,0]) or math.isinf(output_result[i,1]) or math.isinf(output_result[i,2]) or math.isinf(output_result[i,3]) or math.isinf(output_result[i,4]))\
                or (measures_result[i,0] == 0 and measures_result[i,1] == 0 and measures_result[i,2] == 0 and measures_result[i,3] == 0 and measures_result[i,4] == 0 and measures_result[i,5] == 0)\
                or (math.isinf(measures_result[i,0]) or math.isinf(measures_result[i,1]) or math.isinf(measures_result[i,2]) or math.isinf(measures_result[i,3]) or math.isinf(measures_result[i,4]) or math.isinf(measures_result[i,5]))\
                or (output_result[i,1]<1.5 and output_result[i,1]>3)\
                or (output_result[i,2]<0.5 and output_result[i,2]>2)\
                or (output_result[i,3]<0 and output_result[i,3]>1.5)\
                or (not(output_result[i,1]>output_result[i,2] and output_result[i,2]>output_result[i,3]))\
                or (measures_result[i,6]<0.25):
            print(i)
            print("fa = ", measures_result[i,6])
            taille = output_result.shape[0]
            output_result = np.delete(output_result, (i), axis=0)
            measures_result = np.delete(measures_result, (i), axis=0)
            taille2 = output_result.shape[0]
            print("avant/après :", taille,"/",taille2)
            i-=1
        i+=1
        print("i de fin de booucle",i)

    #on supprime la colonne fa qui ne nous sert plus à rien
    measures_result = np.delete(measures_result, 6, 1)


    print("je suis sorti de la boucle")
    np.savetxt("output_result2.csv", output_result, delimiter=",")
    np.savetxt("measure_result2.csv", measures_result, delimiter=",")
    mat_dic = {'dki': measures_result,
               'wmti_paras': output_result}

    savemat("datasets/wmti_dki_constraint_test.mat", mat_dic)

    return True

def load_nifti_img(filepath, dtype=np.float64):
    '''
    NIFTI Image Loader
    :param filepath: path to the input NIFTI image
    :param dtype: dataio type of the nifti numpy array
    :return: return numpy array
    '''
    nim = nib.load(filepath)
    out_nii_array = np.array(nim.get_fdata(),dtype=dtype)
    out_nii_array = np.squeeze(out_nii_array) # drop singleton dim in case temporal dim exists
    meta = {'affine': nim.affine,
            'dim': nim.header['dim'],
            'pixdim': nim.header['pixdim'],
            'name': os.path.basename(filepath)
            }

    return out_nii_array, meta

def show_slices(slices):

   """ Function to display row of image slices """

   fig, axes = plt.subplots(1, len(slices))
   for i, slice in enumerate(slices):
       axes[i].imshow(slice.T, cmap="gray", origin="lower")



def convert_nii_in_array(f_nii_path, Da_nii_path, Depar_nii_path, Deperp_nii_path, kappa_nii_path,
                         ad_nii_path, rd_nii_path, md_nii_path, ak_nii_path, rk_nii_path, mk_nii_path, nb_params=5, nb_measures=6):
    """

    :return: array of size (nb of parameters x 5) corresponding of the parameters
             array of size (nb of measures x 6) corresponding of the measures
    """
    f_array, meta_f = load_nifti_img(f_nii_path, dtype=np.float64)
    Da_array, meta_Da = load_nifti_img(Da_nii_path, dtype=np.float64)
    Depar_array, meta_Depar = load_nifti_img(Depar_nii_path, dtype=np.float64)
    Deperp_array, meta_Deperp = load_nifti_img(Deperp_nii_path, dtype=np.float64)
    kappa_array, meta_kappa = load_nifti_img(kappa_nii_path, dtype=np.float64)
    output_array = np.zeros((f_array.shape[0],f_array.shape[1],f_array.shape[2],nb_params))

    ad_array, meta_ad = load_nifti_img(ad_nii_path, dtype=np.float64)
    rd_array, meta_rd = load_nifti_img(rd_nii_path, dtype=np.float64)
    md_array, meta_md = load_nifti_img(md_nii_path, dtype=np.float64)
    ak_array, meta_ak = load_nifti_img(ak_nii_path, dtype=np.float64)
    rk_array, meta_rk = load_nifti_img(rk_nii_path, dtype=np.float64)
    mk_array, meta_mk = load_nifti_img(mk_nii_path, dtype=np.float64)
    measures_array = np.zeros((ad_array.shape[0],ad_array.shape[1],ad_array.shape[2],nb_measures))

    ############################# conversion wmti en array
    for i in range(f_array.shape[2]):
        for j in range(f_array.shape[1]):
            for k in range(f_array.shape[0]):
                output_array[k,j,i,0] = f_array[k,j,i]
                output_array[k,j,i,1] = Da_array[k,j,i]
                output_array[k,j,i,2] = Depar_array[k,j,i]
                output_array[k,j,i,3] = Deperp_array[k,j,i]
                output_array[k,j,i,4] = kappa_array[k,j,i]

    output_result = np.zeros((output_array.shape[0]*output_array.shape[1]*output_array.shape[2],nb_params))  # va contenir les wmti
    t=0
    for i in range(output_array.shape[2]):
        for j in range(output_array.shape[1]):
            for k in range(output_array.shape[0]):
                if not(math.isnan(output_array[k,j,i,0])):
                    output_result[t,0] = output_array[k,j,i,0]
                    output_result[t,1] = output_array[k,j,i,1]
                    output_result[t,2] = output_array[k,j,i,2]
                    output_result[t,3] = output_array[k,j,i,3]
                    output_result[t,4] = output_array[k,j,i,4]
                    t += 1

    ##################### conversion measures en array
    for i in range(ad_array.shape[2]):
        for j in range(ad_array.shape[1]):
            for k in range(ad_array.shape[0]):
                measures_array[k,j,i,0] = md_array[k,j,i]
                measures_array[k,j,i,1] = ad_array[k,j,i]
                measures_array[k,j,i,2] = rd_array[k,j,i]
                measures_array[k,j,i,3] = mk_array[k,j,i]
                measures_array[k,j,i,4] = ak_array[k,j,i]
                measures_array[k,j,i,5] = rk_array[k,j,i]

    measures_result = np.zeros((output_array.shape[0]*output_array.shape[1]*output_array.shape[2],nb_measures))  # va contenir les dki
    t=0
    for i in range(measures_array.shape[2]):
        for j in range(measures_array.shape[1]):
            for k in range(measures_array.shape[0]):
                if not(math.isnan(measures_array[k,j,i,0])):
                    measures_result[t,0] = measures_array[k,j,i,0]
                    measures_result[t,1] = measures_array[k,j,i,1]
                    measures_result[t,2] = measures_array[k,j,i,2]
                    measures_result[t,3] = measures_array[k,j,i,3]
                    measures_result[t,4] = measures_array[k,j,i,4]
                    measures_result[t,5] = measures_array[k,j,i,5]
                    t += 1

    # filtrage
    # for i in range(len(output_result)-1,-1,-1): #on supprimer les zeros et les 'inf' en trop + filtrage des valeurs de Da trop faibles
    for i in range(108364):  #(len(output_result)):  # on supprimer les zeros et les 'inf' en trop + filtrage des valeurs de Da trop faibles
        if (output_result[i,0] == 0 and output_result[i,1] == 0 and output_result[i,2] == 0 and output_result[i,3] == 0 and output_result[i,4] == 0) \
                or (math.isinf(output_result[i,0]) or math.isinf(output_result[i,1]) or math.isinf(output_result[i,2]) or math.isinf(output_result[i,3]) or math.isinf(output_result[i,4]))\
                or (measures_result[i,0] == 0 and measures_result[i,1] == 0 and measures_result[i,2] == 0 and measures_result[i,3] == 0 and measures_result[i,4] == 0 and measures_result[i,5] == 0)\
                or (math.isinf(measures_result[i,0]) or math.isinf(measures_result[i,1]) or math.isinf(measures_result[i,2]) or math.isinf(measures_result[i,3]) or math.isinf(measures_result[i,4]) or math.isinf(measures_result[i,5]))\
                or (output_result[i,1]<1.5 and output_result[i,1]>3)\
                or (output_result[i,2]<0.5 and output_result[i,2]>2)\
                or (output_result[i,3]<0 and output_result[i,3]>1.5)\
                or (not(output_result[i,1]>output_result[i,2] and output_result[i,2]>output_result[i,3])):
            i_flag = i
            output_result = np.delete(output_result, (i), axis=0)
            measures_result = np.delete(measures_result, (i), axis=0)
            print("i/i_flag)", (i, i_flag))

    return output_result, measures_result





def convert_array_in_mat(x_train, y_train):
    mat_dic = {'dki': x_train,
               'wmti_paras': y_train}

    savemat("datasets/wmti_dki_constraint_nii.mat", mat_dic)

    return True

if __name__ == '__main__':

    f_nii_path = '/home/max/Desktop/cibmaitsrv1/Data/20191002_072533_RGR_5113_SANDI_1_1/data_processed/18_D30/WMTI_Watson/f.nii'
    Da_nii_path = '/home/max/Desktop/cibmaitsrv1/Data/20191002_072533_RGR_5113_SANDI_1_1/data_processed/18_D30/WMTI_Watson/Da.nii'
    Depar_nii_path = '/home/max/Desktop/cibmaitsrv1/Data/20191002_072533_RGR_5113_SANDI_1_1/data_processed/18_D30/WMTI_Watson/Depar.nii'
    Deperp_nii_path = '/home/max/Desktop/cibmaitsrv1/Data/20191002_072533_RGR_5113_SANDI_1_1/data_processed/18_D30/WMTI_Watson/Deperp.nii'
    kappa_nii_path = '/home/max/Desktop/cibmaitsrv1/Data/20191002_072533_RGR_5113_SANDI_1_1/data_processed/18_D30/WMTI_Watson/kappa.nii'

    ad = '/home/max/Desktop/cibmaitsrv1/Data/20191002_072533_RGR_5113_SANDI_1_1/data_processed/18_D30/wlls_3shells_dn_rc_gc_r_outliers/ad.nii'
    rd = '/home/max/Desktop/cibmaitsrv1/Data/20191002_072533_RGR_5113_SANDI_1_1/data_processed/18_D30/wlls_3shells_dn_rc_gc_r_outliers/rd.nii'
    md = '/home/max/Desktop/cibmaitsrv1/Data/20191002_072533_RGR_5113_SANDI_1_1/data_processed/18_D30/wlls_3shells_dn_rc_gc_r_outliers/md.nii'
    ak = '/home/max/Desktop/cibmaitsrv1/Data/20191002_072533_RGR_5113_SANDI_1_1/data_processed/18_D30/wlls_3shells_dn_rc_gc_r_outliers/ak.nii'
    rk = '/home/max/Desktop/cibmaitsrv1/Data/20191002_072533_RGR_5113_SANDI_1_1/data_processed/18_D30/wlls_3shells_dn_rc_gc_r_outliers/rk.nii'
    mk = '/home/max/Desktop/cibmaitsrv1/Data/20191002_072533_RGR_5113_SANDI_1_1/data_processed/18_D30/wlls_3shells_dn_rc_gc_r_outliers/mk.nii'


    # output_result, measures_result = convert_nii_in_array(f_nii_path, Da_nii_path, Depar_nii_path, Deperp_nii_path,kappa_nii_path,
    #                                                       ad, rd, md, ak, rk, mk,
    #                                                       nb_params=5,
    #                                                       nb_measures=6)
    #
    # convert_array_in_mat(measures_result, output_result)

    # f_array, meta_f = load_nifti_img(f_nii_path, dtype=np.float64)
    # Da_array, meta_Da = load_nifti_img(Da_nii_path, dtype=np.float64)
    # Depar_array, meta_Depar = load_nifti_img(Depar_nii_path, dtype=np.float64)
    # Deperp_array, meta_Deperp = load_nifti_img(Deperp_nii_path, dtype=np.float64)
    # kappa_array, meta_kappa = load_nifti_img(kappa_nii_path, dtype=np.float64)
    #
    # ad_array, meta_ad = load_nifti_img(ad, dtype=np.float64)
    # rd_array, meta_rd = load_nifti_img(rd, dtype=np.float64)
    # md_array, meta_md = load_nifti_img(md, dtype=np.float64)
    # ak_array, meta_ak = load_nifti_img(ak, dtype=np.float64)
    # rk_array, meta_rk = load_nifti_img(rk, dtype=np.float64)
    # mk_array, meta_mk = load_nifti_img(mk, dtype=np.float64)
    #
    # X = 43
    # Y =43
    # Z = 3
    # print(Da_array[X,Y,Z])
    # print(Da_array.shape)
    #
    # print(meta_ad)
    #
    # img = nib.load(Da_nii_path)
    # print(img.shape)
    # data = img.get_fdata()
    # print(data[X,Y,Z])
    # print(img.header)
    #
    # slice_0 = data[64, :, :]
    # slice_1 = data[:, 48, :]
    # slice_2 = data[:, :, 8]
    # show_slices([slice_0, slice_1, slice_2])
    #
    # plt.suptitle("Center slices for MRI image")
    # plt.show()
    #
    path_mat_matrix = '/home/max/Desktop/cibmaitsrv1/Max_Bourgeat/matrix.mat'
    process_mat_matrix(path_mat_matrix)


