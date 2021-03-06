import os
import time
import math
import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
import torch.utils.data as Data
import torch
from torch.autograd import Variable
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import logging
import logging.config
from matplotlib.ticker import PercentFormatter
from util_dev import device


np.set_printoptions(suppress=True)

def load_model_from_checkpoint(checkpoint_file, model, optimizer, scheduler):
    checkpoint = torch.load(checkpoint_file)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict((checkpoint['opimizer_state_dict']))
    scheduler.load_state_dict((checkpoint['scheduler_state_dict']))
    device = checkpoint['device']
    return checkpoint['epoch'], model, optimizer, scheduler, device

def asMinutes(s):
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)

def timeSince(since, percent):
    now = time.time()
    s = now - since
    es = s / (percent)
    rs = es - s
    return '%s (- %s)' % (asMinutes(s), asMinutes(rs))	

def showPlot(points_lists, save_dir='', fig_name='losses.png'):
    num_point_arrays = len(points_lists)
    fig, axs = plt.subplots(num_point_arrays, 1)
    for i, ax in enumerate(axs.flatten()):
        ax.plot(points_lists[i])

    if os.path.isdir(save_dir): 
        plt.savefig(os.path.join(save_dir, f"{fig_name}")) 

    plt.show()
    #plt.close()

def data_normalize(input_array, min_val, max_val, multipler=1, inverse_transform=False):
    x = input_array.reshape(-1, 1)
    if inverse_transform:
        y = x/multipler * (max_val - min_val) + min_val
    else:
        y = multipler*(x - min_val)/(max_val - min_val)
    return y

#f = [0,1], Da = [1.5 3], Depar = [0.5 2], Deperp = [0 1.5], kappa = [2 128]
def min_max_scale(input_data, ranges=[[0,1], [1.5, 3], [0.5, 2], [0, 1.5], [2, 128]], multipler=1, inverse_transform=False):
    #input_data: 2D numpy array, nSamples x nVariables
    assert input_data.shape[1] == len(ranges)
    input_scaled = np.zeros(input_data.shape)

    for i in range(len(ranges)):
        input_scaled[:, i] = data_normalize(input_data[:, i], *ranges[i], multipler=multipler, inverse_transform=inverse_transform)[:,0]

    return input_scaled.astype(np.float32)

def sk_data_scaler(input_data, scale_type='MinMax', mm_range=(0, 1)):
    #input_data: nSamples x nFeatures
    assert scale_type in ['MinMax', 'Standard'], "Wrong scale_type !!!"
    scaler = MinMaxScaler(feature_range=mm_range) if scale_type is 'MinMax' else StandardScaler()
        
    scaler.fit(input_data)
    normalized = scaler.transform(input_data)

    return normalized.astype(np.float32), scaler

def custom_scale(input_data, scale=[100, 100, 100, 100, 1], b=[0,0,0,0,0], inverse_transform=False):
    #output: y=x*scale + b
    if b==None: b=[0,0,0,0,0]
    b = np.array(b).reshape(1,-1)
    if inverse_transform:
        scaled = (input_data - b) / np.array(scale).reshape(1,-1)
    else:
        scaled = input_data * np.array(scale).reshape(1,-1) + b
    return scaled.astype(np.float32)

def preprocess_datasets(datapath, mat_filename, batch_size=256, train_perc=0.6, val_perc=0.2, data_norm=True, x_scaler=None,
                        device = device, scale_type=0, scale=None, intercept=None, dki_norm_type='MinMax', 
                        wmti_ranges=[[0,1], [1.5, 3], [0.5, 2], [0, 1.5], [2, 128]], seed=9):
    '''
    output: train, val, test datasets
    each: input and target tensor pairs
    exmaple: 
    train[0]: (tensor([ 0.7696,  2.1042,  0.1024,  0.2498,  0.0322, 25.6437]), tensor([  0.9111,   2.1900,   1.4116,   1.0584, 122.4471]))
    '''
    datafile = os.path.join(datapath, mat_filename)
    mats = sio.loadmat(datafile)
    dki = mats['dki'].astype(np.float32)
    wmti = mats['wmti_paras'].astype(np.float32)

    if data_norm:
        #input scaling
        if x_scaler == None:
            dki, x_scaler = sk_data_scaler(dki, scale_type=dki_norm_type)
        else:
            normalized = x_scaler.transform(dki)
            dki = normalized.astype(np.float32)

        #output scaling
        if scale_type==1:
            wmti = custom_scale(wmti, scale, b=intercept) #[100, 100, 100, 100, 1]#[10, 10, 10, 10, 0.1]
        elif scale_type==2:
            wmti = min_max_scale(wmti, multipler=scale, ranges=wmti_ranges)
        else:
            assert scale_type==0, "scale_type must be 0, 1 or 2 !!!"

    x, y = Variable(torch.tensor(dki, device=device)), Variable(torch.tensor(wmti, device=device))

    assert x.shape[0] == y.shape[0]

    torch_dataset = Data.TensorDataset(x, y)
    train_size = int(train_perc * x.shape[0])
    val_size = int(val_perc * x.shape[0])
    test_size = x.shape[0] - train_size - val_size

    print(f"--train_size: {train_size}, val_size: {val_size}, test_size: {test_size}")
    train_data, val_data, test_data = Data.random_split(torch_dataset, [train_size, val_size, test_size], generator=torch.Generator().manual_seed(seed))  

    train_loader = Data.DataLoader(
        dataset=train_data, 
        batch_size=batch_size, 
        shuffle=True)# 

    val_loader = Data.DataLoader(
        dataset=val_data, 
        batch_size=batch_size, 
        shuffle=True)  

    test_loader = Data.DataLoader(
        dataset=test_data, 
        batch_size=batch_size, 
        shuffle=True)  

    return train_loader, val_loader, test_loader, x_scaler   

def setup_log(path, logbsname):
    timestr = time.strftime("%Y%m%d-%H%M%S")
    # create logger
    logger = logging.getLogger(__name__)
    # logger.handlers = []
    logger.propagate = False
    logger.setLevel(logging.DEBUG)
    # create console handler and set level to debug
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    # create formatter
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    # add formatter to ch
    ch.setFormatter(formatter)
    # add ch to logger
    # logger.handlers = []
    logger.addHandler(ch)
    # file handler
    fh = logging.FileHandler(os.path.join(path, f"{logbsname}-{timestr}.log"))
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    return logger    

def preprocess_input_dataset(datapath, mat_filename, batch_size=256, data_norm=True, x_scaler=None, dki_norm_type='MinMax', device = device):
    '''
    '''
    datafile = os.path.join(datapath, mat_filename)
    mats = sio.loadmat(datafile)
    dki = mats['dki'].astype(np.float32)
    dki_r = np.copy(dki)
    if data_norm:
        if x_scaler==None:
            dki, x_scaler = sk_data_scaler(dki, scale_type=dki_norm_type)
        else:
            normalized = x_scaler.transform(dki)
            dki = normalized.astype(np.float32)

    x= Variable(torch.tensor(dki, device=device))

    dataset = Data.TensorDataset(x)

    data_loader = Data.DataLoader(
        dataset=dataset, 
        batch_size=batch_size, 
        shuffle=False)

    return data_loader, dki_r, x_scaler

# def affichage(array, datapath, mat_filename):
#
#     datafile = os.path.join(datapath, mat_filename)
#     mats = sio.loadmat(datafile)
#     # dki = mats['dki'].astype(np.float32)
#     wmti = mats['wmti_paras'].astype(np.float32)
#     print("wmti" , wmti, "len :",len(wmti))
#
#     prediction_f = array[:,0,1]
#     y_train_f = array[:,0,0]
#     # print("prediction:",prediction_f, "type:",type(prediction_f))
#
#     prediction_Da = array[:,1,1]
#     y_train_Da = array[:,1,0]
#     # print("prediction:",prediction_Da)
#
#     prediction_Depar = array[:,2,1]
#     y_train_Depar = array[:,2,0]
#     # print("prediction:",prediction_Depar)
#
#     prediction_Deperp = array[:,3,1]
#     y_train_Deperp = array[:,3,0]
#     # print("prediction:",prediction_Deperp)
#
#     prediction_kappa = array[:,4,1]
#     y_train_kappa = array[:,4,0]
#     # print("prediction:",prediction_kappa)
#
#     prediction = array[:,:,1]
#     print("prediction:",prediction, "len :",len(prediction))
#
#     x1 = np.linspace(0, 1, 2000)
#     x2 = np.linspace(1.5, 3, 2000)
#     x22 = np.linspace(0.5, 2, 2000)
#     x23 = np.linspace(0, 1.5, 2000)
#     x3 = np.linspace(0, 129, 2000)
#
#     f, axes = plt.subplots(nrows=2, ncols=3, sharex=False, sharey=False)
#
#     axes[0][0].scatter(y_train_f, prediction_f, marker=".", s=5)
#     axes[0][0].set_ylabel('prediction')
#     axes[0][0].set_xlabel('ground truth')
#     axes[0][0].set_title('f')
#     axes[0][0].plot(x1, x1, 'r')
#     axes[0][0].plot(x1, x1 - 0.1 * x1, 'r', linestyle='dashed')
#     axes[0][0].plot(x1, x1 + 0.1 * x1, 'r', linestyle='dashed')
#     axes[0][0].set_xlim(-0.1, 1.1)
#
#     axes[0][1].scatter(y_train_kappa, prediction_kappa, marker=".", s=5)
#     axes[0][1].set_ylabel('prediction')
#     axes[0][1].set_xlabel('ground truth')
#     axes[0][1].set_title('kappa')
#     axes[0][1].plot(x3, x3, 'r')
#     axes[0][1].plot(x3, x3 - 0.1 * x3, 'r', linestyle='dashed')
#     axes[0][1].plot(x3, x3 + 0.1 * x3, 'r', linestyle='dashed')
#     axes[0][1].set_xlim(-0.1, 42.1)
#
#     axes[1][0].scatter(y_train_Da, prediction_Da, marker='.', s=5)
#     axes[1][0].set_ylabel('prediction')
#     axes[1][0].set_xlabel('ground truth')
#     axes[1][0].set_title('Da')
#     axes[1][0].plot(x2, x2, 'r')
#     axes[1][0].plot(x2, x2 - 0.1 * x2, 'r', linestyle='dashed')
#     axes[1][0].plot(x2, x2 + 0.1 * x2, 'r', linestyle='dashed')
#     axes[1][0].set_xlim(1.4, 3.1)
#
#     axes[1][1].scatter(y_train_Depar, prediction_Depar, marker='.', s=5)
#     axes[1][1].set_ylabel('prediction')
#     axes[1][1].set_xlabel('ground truth')
#     axes[1][1].set_title('Depar')
#     axes[1][1].plot(x22, x22, 'r')
#     axes[1][1].plot(x22, x22 - 0.1 * x22, 'r', linestyle='dashed')
#     axes[1][1].plot(x22, x22 + 0.1 * x22, 'r', linestyle='dashed')
#     axes[1][1].set_xlim(0.4,2.1)
#
#     axes[0][2].scatter(y_train_Deperp, prediction_Deperp, marker='.', s=5)
#     axes[0][2].set_ylabel('prediction')
#     axes[0][2].set_xlabel('ground truth')
#     axes[0][2].set_title('Deperp')
#     axes[0][2].plot(x23, x23, 'r')
#     axes[0][2].plot(x23, x23 - 0.1 * x23, 'r', linestyle='dashed')
#     axes[0][2].plot(x23, x23 + 0.1 * x23, 'r', linestyle='dashed')
#     axes[0][2].set_xlim(-0.1, 1.6)
#
#     plt.suptitle('Estimation with LSTM + attention model ', fontsize=15)
#
#     f.tight_layout()
#     plt.show()
#
#     plt.show()
#
#     prediction = array[:,:,1]
#     train = array[:,:,1]
#
#     errors = np.abs((prediction - wmti)) / wmti * 100 + 1e-15  # in %
#     perc = 5  # %
#     nsamples = errors.shape[0]
#     print(f"nSamples: {nsamples}")
#     # savedir = '/home/bourgeat/PycharmProjects/EPFL/hist'
#     # savetitle = 'Error_dist'
#     plotparam = ['f', 'Da', 'Depar', 'Deperp', 'kappa']
#     bins_range = (1e-10, 1e2)  # %
#     nbins = 300
#     fig, axes = plt.subplots(2, len(plotparam) // 2 + 1, figsize=(15, 8))
#     plt.suptitle('relative-error distribution')
#     for ii, par in enumerate(plotparam):
#         ax = axes.flatten()[ii]
#         logbins = np.logspace(np.log10(bins_range[0]), np.log10(bins_range[-1]), nbins)
#         logbins = np.unique(np.round(logbins, 3))
#         hist_, _, _ = ax.hist(errors[:, ii], weights=np.ones(nsamples) / nsamples, bins=logbins)
#         idx_less_than = np.sum(logbins <= perc) - 1  # error <= perc %, idx in logbins
#         perc_ = logbins[idx_less_than]
#         perc_lt = round(np.sum(hist_[0:idx_less_than]) * 100, 2)  # %,
#         tex = f"Portion(error <= {perc_}%): {perc_lt}%"
#         ax.set_xlim(1e-3, 1e2)
#         ax.set_xscale('log')
#         ax.set_xlabel(f'error (%) \n {tex}')
#         ax.set_ylabel('Percentage')
#         ax.set_title(par)
#         ax.yaxis.set_major_formatter(PercentFormatter(1))
#     plt.tight_layout()
#     plt.show()




plotparam = ['f', 'Da', 'Depar', 'Deperp','kappa']#
lim = [[0,1], [1.5, 3], [0.5, 2], [0,1.5], [2, 128]]
def plot_scatter(target, pred, plot_err=0.1, plotparam=plotparam, lim=lim, title = None, savetitle = None, savedir = None):
    plt.figure(figsize=(30,7))
    plt.suptitle(title)
    for i, string in enumerate(plotparam):
        plt.subplot(1,len(plotparam),i+1)
        plt.scatter(target[:,i], pred[:,i], s=.5)
        llim, hlim = lim[i][0], lim[i][1]
        x = np.linspace(llim, hlim)
        plt.plot(x,x, 'r', linewidth = 1)
        plt.plot(x,x*(1+plot_err), 'k--', linewidth = 1)
        plt.plot(x,x*(1-plot_err), 'k--', linewidth = 1)
        temp = (hlim-llim)/10
        plt.xlim(llim - temp, hlim + temp), plt.ylim(llim - temp, hlim + temp)
        plt.xlabel('target'), plt.ylabel('prediction')
        plt.title(string)
    if not(savetitle == None): plt.savefig(f"{savedir}/{savetitle}.png")
    plt.show()

def affichage():
    testdatapath = '/home/max/PycharmProjects/DL_estimation2/output'
    testdatafile = f'{testdatapath}/test_pred_tgt.npy'
    pred_tgt = np.load(testdatafile)
    pred = pred_tgt[:,:,0]
    target = pred_tgt[:,:,1]
    step = 2
    pred_ds = pred[::step, :]
    target_ds = target[::step, :]
    print(f"numOfSamples: {pred_ds.shape[0]}")
    plot_scatter(target_ds, pred_ds, title='target-vs-prediction', savetitle='PD_vs_GT', savedir=testdatapath)


    errors = np.abs((pred - target))/target * 100 + 1e-15 #in %
    perc = 5 #%
    nsamples = errors.shape[0]
    print(f"nSamples: {nsamples}")
    savedir = testdatapath
    savetitle = 'Error_dist'
    bins_range = (1e-10, 1e2) #%
    nbins = 300
    fig, axes = plt.subplots(1,len(plotparam),figsize=(30,6))
    plt.suptitle('relative-error distribution')
    for ii, par in enumerate(plotparam):
        ax = axes.flatten()[ii] if len(plotparam)>1 else axes
        # logbins = np.logspace(np.log10(bins[0]),np.log10(bins[-1]),len(bins))
        logbins = np.logspace(np.log10(bins_range[0]),np.log10(bins_range[-1]),nbins)
        logbins = np.unique(np.round(logbins,3))
        hist_, _, _ = ax.hist(errors[:, ii], weights=np.ones(nsamples)/nsamples, bins=logbins)
        idx_less_than = np.sum(logbins<=perc)-1 #error <= perc %, idx in logbins
        perc_ = logbins[idx_less_than]
        perc_lt = round(np.sum(hist_[0:idx_less_than]) * 100, 2) #%,
        tex = f"Portion(error <= {perc_}%): {perc_lt}%"
        ax.set_xlim(1e-3, 1e2)
        ax.set_xscale('log')
        ax.set_xlabel(f'error (%) \n {tex}')
        ax.set_ylabel('Percentage')
        ax.set_title(par)
        #ax.text(0, 1, tex, fontsize=10, va='top')
        ax.yaxis.set_major_formatter(PercentFormatter(1))
    plt.tight_layout()
    if not(savetitle == None): plt.savefig(f"{savedir}/{savetitle}.png")
    plt.show()