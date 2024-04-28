import os
from easydict import EasyDict as edict
import time
import torch

# init
__C = edict()
cfg = __C
__C.DRL_BATCH = 6
__C.LINEAR_DIM = 1024
__C.SAM_DEPTH = 24
__C.SAM_INDEX = [5, 11, 17, 23]
__C.SAM_NUM_HEADS = 16
#------------------------------TRAIN------------------------
__C.SEED =1012 # random seed,  for reproduction
__C.DATASET = 'SHHA' # dataset selection: GCC, SHHA, SHHB, UCF50, QNRF, WE, Mall, UCSD, SHTRGBD, CARPK, PUCPR, RSOC,RSOCLARGEVEHICLE, RSOCSMALLVEHICLE, RSOCSHIP,RSOCSHIPORG

if __C.DATASET == 'UCF50':# only for UCF50
	from datasets.UCF50.setting import cfg_data
	__C.VAL_INDEX = cfg_data.VAL_INDEX 

if __C.DATASET == 'GCC':# only for GCC
	from datasets.GCC.setting import cfg_data
	__C.VAL_MODE = cfg_data.VAL_MODE 


__C.NET = 'ALLMPF' # net selection: MCNN, AlexNet, VGG, VGG_DECODER, Res50, CSRNet, SANet, DDIPMN

__C.PRE_GCC = False # use the pretrained model on GCC dataset
__C.PRE_GCC_MODEL = 'path to model' # path to model

__C.VISUALIZE = True
__C.BN = True
__C.PRETRAINED =False
# contine training
__C.VISUALIZATION = False
__C.RESUME =False
__C.RESUME_PATH = '/data2/haojie/CODE/ALLMPF/exp/04-21_12-38_SHHA_ALLMPF_0.0005/latest_state.pth' # 
__C.RESUME_BEST =False
# __C.RESUME_BEST_PATH = '/data/haojie/CODE/EXTRemoteCC/exp/12-01_22-54_PUCPR_OANet_0.0001/all_ep_1_mae_9.09_mse_9.86.pth'
__C.RESUME_BEST_PATH = '/data2/haojie/CODE/ALLMPF/exp/04-21_15-54_SHHA_ALLMPF_0.0005/all_ep_164_mae_65.31_mse_97.57.pth'

# __C.RESUME_PATH = './epoch199.pth' # 

__C.GPU_ID = [1,2] # sigle gpu: [0], [1] ...; multi gpus: [0,1]
__C.MAIN_GPU = [1]
# learning rate settings
__C.LR =1e-4# learning rate 
__C.LR_DECAY = 0.995# decay rate
__C.LR_DECAY_START = -1 # when training epoch is more than it, the learning rate will be begin to decay
__C.NUM_EPOCH_LR_DECAY = 1 # decay frequency
__C.MAX_EPOCH = 700

# multi-task learning weights, no use for single model, such as MCNN, VGG, VGG_DECODER, Res50, CSRNet, and so on

__C.LAMBDA_1 = 1e-3# SANet:0.001 CMTL 0.0001
                                                               

# print 
__C.PRINT_FREQ = 10

now = time.strftime("%m-%d_%H-%M", time.localtime())

__C.EXP_NAME = now \
			 + '_' + __C.DATASET \
             + '_' + __C.NET \
             + '_' + str(__C.LR)

if __C.DATASET == 'UCF50':
	__C.EXP_NAME += '_' + str(__C.VAL_INDEX)	

if __C.DATASET == 'GCC':
	__C.EXP_NAME += '_' + __C.VAL_MODE	

__C.EXP_PATH = './exp' # the path of logs, checkpoints, and current codes


#------------------------------VAL------------------------
__C.VAL_DENSE_START = 2
__C.VAL_FREQ = 1 # Before __C.VAL_DENSE_START epoches, the freq is set as __C.VAL_FREQ

#------------------------------VIS------------------------
__C.VISIBLE_NUM_IMGS = 1 #  must be 1 for training images with the different sizes



#================================================================================
#================================================================================
#================================================================================  
