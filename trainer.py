import os

import numpy as np
from scipy.io import savemat
from skimage.metrics import structural_similarity as ssim

from torch import optim
from torch.autograd import Variable
from torch.optim.lr_scheduler import StepLR, CosineAnnealingLR
import torch.nn.functional as F
from models.CC import CrowdCounter
from collections import OrderedDict
from models.SCC_Model.DRL import DRL as DRLnet
from models.SCC_Model.DRL import ExBuffer, bufferLoader
from torch.utils.data import DataLoader
from models.SCC_Model.ALLMPF import Build_Train_net
# from datasets.SHHB.setting import cfg_data
import cv2
import time
from config import cfg
from misc.utils import *
from tqdm import tqdm, trange
import pdb
from visdom import Visdom
import matplotlib.pyplot as plt
import copy
import collections
vis = Visdom(env='main', port=4000)
vis.line([[0.]], [0], win='q-value-loss', opts=dict(title='q-value-loss', legend=['train']))
vis.line([[0.]], [0], win='train_loss', opts=dict(title='train_loss', legend=['train']))
vis.line([[0.]], [0], win='dm_loss', opts=dict(title='dm_loss', legend=['train']))
vis.line([[0.]], [0], win='oa_loss', opts=dict(title='oa_loss', legend=['train']))

# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
class Trainer():
    def __init__(self, drldataloader, dataloader, cfg_data, pwd):

        self.cfg_data = cfg_data

        self.data_mode = cfg.DATASET
        self.exp_name = cfg.EXP_NAME
        self.exp_path = cfg.EXP_PATH
        self.pwd = pwd 
        self.net_name = cfg.NET
        self.net = CrowdCounter(cfg.GPU_ID,self.net_name)
        self.DRLnet = DRLnet(blocks=cfg.SAM_DEPTH//2, cfg_data = self.cfg_data)  #TODO
        self.buffer = ExBuffer(50)
        self.block_depth = cfg.SAM_DEPTH
        self.detachable_num = cfg.SAM_DEPTH//2
        self.epoch_record = []
        self.remian_blocks = list(range(self.block_depth))
        #初始化参数池
        self.buffer.clean()
        self.buffer_criterion = torch.nn.MSELoss()
        # 优化器
        self.buffer_optimizer = torch.optim.Adam(
            self.DRLnet.compute_action.parameters(),
            lr=1e-5,
            weight_decay=5e-2)
        self.trainFlag = False
        self.valFlag = False
        self.training = True
        # params_decay, params_no_decay=self.split_parameters(self.net.CCN)
        self.optimizer = optim.AdamW(self.net.CCN.parameters(), lr=cfg.LR, weight_decay=1e-7)
        # self.optimizer = optim.Adam(self.net.CCN.parameters(), lr=cfg.LR, weight_decay=1e-3)
        # self.optimizer.add_param_group({'params':params_bias})
        # self.optimizer = optim.SGD(self.net.CCN.parameters(), cfg.LR, momentum=0.95,weight_decay=0.0001)
        # self.optimizer.add_param_group({'params':params_decay, 'weight_decay': 5e-4})
        # del params_decay, params_no_decay
        self.scheduler = StepLR(self.optimizer, step_size=cfg.NUM_EPOCH_LR_DECAY, gamma=cfg.LR_DECAY)          
        # self.scheduler = CosineAnnealingLR(optimizer=self.optimizer, T_max=120, eta_min=1e-6)
        self.train_record = {'best_mae': 1e20, 'best_mse':1e20, 'best_model_name': ''}
        self.timer = {'iter time' : Timer(),'train time' : Timer(),'val time' : Timer()} 
        self.train_buffer_count = 0
        self.epoch = 0
        self.i_tb = 0
        
        if cfg.PRE_GCC:
            self.net.load_state_dict(torch.load(cfg.PRE_GCC_MODEL))

        self.train_loader, self.val_loader, self.restore_transform = dataloader()
        self.drl_train_loader, self.drl_val_loader, self.restore_transform = drldataloader()
        self.instancenorm = nn.InstanceNorm2d(num_features=1)
        if cfg.RESUME:
            latest_state = torch.load(cfg.RESUME_PATH)
            self.net.load_state_dict(latest_state['net'])
            DRL_state_dict = torch.load(os.path.join(cfg.RESUME_PATH, '../code/log/compute_action.pth'))
            self.DRLnet.compute_action.load_state_dict(DRL_state_dict)
            self.optimizer.load_state_dict(latest_state['optimizer'])
            self.scheduler.load_state_dict(latest_state['scheduler'])
            self.epoch = latest_state['epoch'] + 1
            self.i_tb = latest_state['i_tb']
            self.train_record = latest_state['train_record']
            self.exp_path = latest_state['exp_path']
            self.exp_name = latest_state['exp_name']
        if cfg.RESUME_BEST:
            # latest_state = torch.load(cfg.RESUME_BEST_PATH)
            # new_state_dict = OrderedDict()
            # for k,v in latest_state.items():  # single GPU load multi-GPU pre-trained model
            #     name = k.split(".")
            #     name.remove('module')
            #     name = ".".join(name)
            #     new_state_dict[name] = v 
            # self.net.load_state_dict(new_state_dict)
            latest_state = torch.load(cfg.RESUME_BEST_PATH)
            self.net.load_state_dict(latest_state)
        self.writer, self.log_txt = logger(self.exp_path, self.exp_name, self.pwd, 'exp', resume=cfg.RESUME)
        self.file = open(self.exp_path+'/' + self.exp_name+'/code/log/' + 'training.txt', 'a')
    def initialize_weights(self, net):
        for m in net.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    def build_train_net(self, detachable_list):
        new_encoder = Build_Train_net(detachable_list)
        return new_encoder
    def forward(self):
        if not os.path.exists(self.cfg_data.LOG_DIR):
            os.mkdir(self.cfg_data.LOG_DIR)
        file = open(self.cfg_data.LOG_DIR+'training.txt', 'w').close()
        # self.validate_V3()
        best_mae, best_epoch = 1000.0, 0
        for epoch in range(self.epoch,cfg.MAX_EPOCH):
            print("**********************Learing rate in this epoch is {}***********************".format(self.optimizer.param_groups[0]['lr']))
            if cfg.DATASET=='UCF50':
                print('epoch: ', epoch)
            self.epoch = epoch

            # training    
            self.timer['train time'].tic()
            # with torch.no_grad():
                # self.validate_V2()
                # self.validate_V4()
            # self.update_buffer()
            # masks =self.predict_buffer()
            # new_net = self.build_train_net(masks)
            if self.epoch<=15 or self.training:
                    self.train()
            if self.epoch>15 and not self.valFlag:
            # if not self.valFlag:
                    self.update_buffer()
            if self.trainFlag:
                    self.train_buffer()
                    torch.cuda.empty_cache()
                    masks =self.predict_buffer() 
                    torch.save(self.DRLnet.compute_action.state_dict(),os.path.join(self.exp_path, self.exp_name, 'code/log', 'compute_action.pth'))
                    torch.cuda.empty_cache()
            if self.valFlag and not self.training:
                    new_net = self.build_train_net(masks)
                    new_net = torch.nn.DataParallel(new_net, device_ids=cfg.GPU_ID).cuda()
                    self.optimizer = optim.AdamW(new_net.parameters(), lr=cfg.LR, weight_decay=1e-7)
                    self.scheduler = StepLR(self.optimizer, step_size=cfg.NUM_EPOCH_LR_DECAY, gamma=cfg.LR_DECAY)  
                    with open(os.path.join(self.exp_path, self.exp_name, 'code/log', 'new_net.txt'), 'a') as f:
                        # 将print函数的输出重定向到文件对象f
                        print(new_net, file=f)
                    # self.initialize_weights(new_net.module.Decoder)
                    self.net.CCN = new_net
                    # self.net.cuda()
                    self.training = True

                
            # else:
            # self.train()
            if epoch > cfg.LR_DECAY_START:
                self.scheduler.step()
            # torch.save(self.net.state_dict(), "./epoch{}.pth".format(epoch))
            self.timer['train time'].toc(average=False)

            print( 'train time: {:.2f}s'.format(self.timer['train time'].diff) )
            print( '='*20 )

            # validation
            if (epoch+1)%cfg.VAL_FREQ==0 or epoch>cfg.VAL_DENSE_START:
                if self.valFlag == True:
                    self.timer['val time'].tic()
                    if self.data_mode in ['SHHA', 'SHHB', 'QNRF', 'UCF50', 'SHTRGBD', 'CARPK', 'PUCPR', 'RSOC','RSOCSHIP']:
                        with torch.no_grad():
                            self.validate_V1()
                elif self.data_mode in [ 'RSOCSMALLVEHICLE', 'RSOCSHIPORG', 'RSOCLARGEVEHICLE']:
                    with torch.no_grad():
                        val_mae = self.validate_V4()
                        # if val_mae<best_mae:
                        #     best_mae = val_mae
                        #     best_epoch = self.epoch
                        # if self.epoch-best_epoch>=50:
                        #     self.optimizer.param_groups[0]['lr']*=0.1
                        #     best_epoch = self.epoch
                        print("**********************Learing rate has been decreased to {}***********************".format(self.optimizer.param_groups[0]['lr']))
                elif self.data_mode in [ 'VISDRONE']:
                    with torch.no_grad():
                        self.validate_V2()
                elif self.data_mode in ['RSOCSHIPORG']:
                    with torch.no_grad():
                        self.validate_V5()
                self.timer['val time'].toc(average=False)
                print( 'val time: {:.2f}s'.format(self.timer['val time'].diff) )
    def adaptive_training(self, cut_net):
        optimizer = optim.AdamW(cut_net.parameters(), lr=0.0001, weight_decay=1e-7)
        scheduler = StepLR(optimizer, step_size=cfg.NUM_EPOCH_LR_DECAY, gamma=0.995) 
        loss_mse_fn = nn.MSELoss().cuda()
        cut_net.train()
        cut_net.cuda()
        # block_1_param = cut_net.state_dict()['module.image_encoder.blocks.1.attn.qkv.weight']
        # ada_param = cut_net.state_dict()['module.image_encoder.blocks.0.0.weight']
        # print(block_1_param)
        # print(ada_param)
        # block_count = 0
        # block_name = []
        # for block in cut_net.module.image_encoder.blocks:
        #     for name, param in block.named_parameters():
        #         if param.requires_grad:
        #             # print('module.image_encoder.blocks.'+str(block_count)+'.'+name)
        #             if block_count in block_name:
        #                 continue
        #             else:
        #                 block_name.append(block_count)
        #     block_count+=1
        # self.file.write("当前未被冻结的block层是：{}\n".format(block_name))
        # print("当前未被冻结的block层是：{}".format(block_name))
        # block_name = []


        for epoch in trange(5):
            for i, data in enumerate(self.drl_train_loader, 0):
                img, gt_map, fname  = data
                gt_map = gt_map.float()
                img = Variable(img).cuda()
                gt_map = Variable(gt_map).cuda()
                optimizer.zero_grad()
                
                pred_map= cut_net(img)
                loss = loss_mse_fn(pred_map.squeeze(), gt_map.squeeze())
                
                loss.backward()
                optimizer.step()
            scheduler.step()
        # del img, gt_map, loss
        
        torch.cuda.empty_cache()
        # block_1_param = cut_net.state_dict()['module.image_encoder.blocks.1.attn.qkv.weight']
        # ada_param = cut_net.state_dict()['module.image_encoder.blocks.0.0.weight']
        # print(block_1_param)
        # print(ada_param)
        cut_net.eval()
        return cut_net
    def update_buffer(self, ):
        buf = collections.namedtuple('buf', field_names=['fea', 'Q'])
        self.net.eval()
        # self.net = self.net.cpu()
        #初始化
        input_net = copy.deepcopy(self.net.CCN)
        self.file.write("*************记忆池开始更新*************\n")
        print("*************记忆池开始更新*************")
        for epoch in range(50):#TODO
            print("记忆池蓄水中，当前是第{}个epoch.".format(epoch))
            features_before_cut = 0.0
            features_after_cut = 0.0
            cut_net = self.DRLnet.CUTNET(input_net)
            cut_net = self.adaptive_training(cut_net)

            torch.cuda.empty_cache()
            input_net = input_net.cuda()
            with torch.no_grad():
                for i, data in enumerate(self.drl_train_loader, 0):
                    img, gt_map, fname  = data
                    # gt_map = gt_map.float()
                    img = Variable(img).cuda()
                    # gt_map = Variable(gt_map).cuda()
                    features_before_cut += input_net(img)
                    features_after_cut  += cut_net(img)

            input_net = input_net.cpu()
            cut_net = cut_net.cpu()
            torch.cuda.empty_cache()

            self.DRLnet.compute_action.cuda().eval()
            features_before_cut = self.instancenorm(features_before_cut)
            features_after_cut = self.instancenorm(features_after_cut)
            qval = self.DRLnet.compute_action(features_before_cut, features_after_cut) #0 not cut, 1 cut
            action = np.argmax(qval.data.detach().cpu().numpy())
            reward, _ = self.DRLnet.compute_reward(features_before_cut, features_after_cut, action)
            self.file.write("当前epoch奖励为:{}\n".format(reward))
            print("当前epoch奖励为:{}".format(reward))
            self.DRLnet.compute_action =  self.DRLnet.compute_action.cpu()
            torch.cuda.empty_cache()

            if action == 1 and reward>0:
                temp = reward
                feature_before_cut_newstate = features_before_cut
                feature_after_cut_newstate = features_after_cut
                # # 获取剪枝的掩膜
                # masks = {}
                # for name, module in cut_net.named_modules():
                #     masks[name + '.weight_mask'] = module.weight_mask.clone().detach()
            else:
                cut_net_next = self.DRLnet.CUTNET(input_net)
                cut_net_next = self.adaptive_training(cut_net_next)

                torch.cuda.empty_cache()

                feature_before_cut_newstate, feature_after_cut_newstate = 0.0, 0.0
                input_net = input_net.cuda()
                with torch.no_grad():
                    for i, data in enumerate(self.drl_train_loader, 0):
                        img, gt_map, fname  = data
                        # gt_map = gt_map.float()
                        img = Variable(img).cuda()
                        # gt_map = Variable(gt_map).cuda()
                        feature_before_cut_newstate += input_net(img)
                        feature_after_cut_newstate  += cut_net_next(img)
                
                    input_net = input_net.cpu()
                    cut_net_next = cut_net_next.cpu()
                    torch.cuda.empty_cache()

                    self.DRLnet.compute_action = self.DRLnet.compute_action.cuda()
                    feature_before_cut_newstate = self.instancenorm(feature_before_cut_newstate)
                    feature_after_cut_newstate = self.instancenorm(feature_after_cut_newstate)
                    temp = self.DRLnet.compute_action(feature_before_cut_newstate, feature_after_cut_newstate)
                    temp = np.argmax(temp.data.detach().cpu().numpy())
                    temp_reward, _ = self.DRLnet.compute_reward(feature_before_cut_newstate, feature_after_cut_newstate, temp)
                    temp = reward + 0.90 * temp_reward
                    self.file.write("贪婪策略奖励为:{}\n".format(temp))
                    print("贪婪策略奖励为:{}".format(temp))
                    self.DRLnet.compute_action =  self.DRLnet.compute_action.cpu()
                    torch.cuda.empty_cache()

            qval[0][action] = temp
            if self.buffer.ready2train:
                self.trainFlag = True
                self.file.write("*************记忆池更新完毕*************\n")
                print("*************记忆池更新完毕*************")
                return 
            else:
                temp = buf((feature_before_cut_newstate.detach().cpu(), feature_after_cut_newstate.detach().cpu()), qval.squeeze(0).detach().cpu())
                self.buffer.append(temp)
        self.trainFlag = True
        return
    def train_buffer(self,):
        self.DRLnet.compute_action.cuda().train()
        dataset = bufferLoader(self.buffer.buffer)
        loader = DataLoader(dataset, num_workers=0, batch_size=1, shuffle=True)
        for epoch in range(50):
            self.file.write("*****************训练DRL分类器阶段，共10个epoch，当前epoch：{}*****************\n".format(epoch))
            print("*****************训练DRL分类器阶段，共10个epoch，当前epoch：{}*****************".format(epoch))
            vis_loss= 0.0
            iter_count = 0
            for fea, Q in tqdm(loader):
                # fea =torch.from_numpy(fea)
                feature_before_cut, feature_after_cut = fea[0], fea[1]
                self.buffer_optimizer.zero_grad()
                feature_before_cut, feature_after_cut, Q = Variable(feature_before_cut.squeeze(0)).cuda(), Variable(feature_after_cut.squeeze(0)).cuda(), Q.cuda()
                action_out = self.DRLnet.compute_action(feature_before_cut, feature_after_cut)
                loss = self.buffer_criterion(action_out, Q)
                vis_loss+=loss.item()
                iter_count+=1
                
                loss.backward()
                self.buffer_optimizer.step()
            self.train_buffer_count+=1
            vis.line([[vis_loss/iter_count]], [self.train_buffer_count], win='q-value-loss', update='append')
        self.DRLnet.compute_action.cpu().eval()
        # self.save_train()

    def predict_buffer(self):
        self.DRLnet.compute_action.eval()
        #with torch.no_grad():
        input_net = copy.deepcopy(self.net.CCN)
        cut_net = self.DRLnet.CUTNET(input_net)
        masks = self.collect_pruning_masks(cut_net.module.image_encoder.blocks)
        features_before_cut = 0.0
        features_after_cut = 0.0
        self.DRLnet.compute_action.cuda()
        score_dict = {}
        for epoch in range(50):
            self.file.write("*****************迭代测试DRL阶段，当前epoch：{}，总epoch数为{}*****************\n".format(epoch, 50))
            print("*****************迭代测试DRL阶段，当前epoch：{}，总epoch数为{}*****************".format(epoch, 50))
            cut_net = self.DRLnet.CUTNET(input_net)
            cut_net = self.adaptive_training(cut_net)
            
            with torch.no_grad():
                input_net=input_net.cuda()
                cut_net=cut_net.cuda()
                for i, data in enumerate(self.drl_train_loader, 0):
                    img, gt_map, fname  = data
                    # gt_map = gt_map.float()
                    img = Variable(img).cuda()
                    # gt_map = Variable(gt_map).cuda()
                    features_before_cut += input_net(img)
                    features_after_cut  += cut_net(img)
                features_before_cut = self.instancenorm(features_before_cut)
                features_after_cut = self.instancenorm(features_after_cut)
                qval = self.DRLnet.compute_action(features_before_cut, features_after_cut) #0 not cut, 1 cut
            action = np.argmax(qval.data.detach().cpu().numpy())
            reward, score = self.DRLnet.compute_reward(features_before_cut, features_after_cut, action)
            if action == 1 :
                masks = self.collect_pruning_masks(cut_net.module.image_encoder.blocks)
                score_dict[score] = masks
            if action == 1 and reward > 0:
                self.trainFlag = False 
                self.DRLnet.compute_action.cpu() 
                self.valFlag = True
                # 获取剪枝的掩膜
                masks = self.collect_pruning_masks(cut_net.module.image_encoder.blocks)
                return masks
        if len(score_dict)!=0:
            max_key = max(score_dict.keys())
            masks = score_dict[max_key]
            self.trainFlag = False 
            self.DRLnet.compute_action.cpu() 
            self.valFlag = True
            return masks
        else:
            return masks
    def collect_pruning_masks(self, model, masks=None, parent_name=''):
        """
        Collect all pruning masks from a pruned model into a dictionary.
        
        Args:
        model (torch.nn.Module): The model from which to collect masks.

        Returns:
        dict: A dictionary containing all pruning masks, keyed by module and parameter name.
        """
        masks = {}
        for name, buffer in model.named_buffers():
            if name.endswith('_mask'):
                # 获取掩膜并存储在字典中
                masks[name] = buffer.clone().detach()
        return masks
        
    def train(self): # training for all datasets
        self.net.train()
        mae, dm_losses, oa_losses, r_loss, d_loss = 0.0, 0.0, 0.0, 0.0, 0.0
        losses = 0.0
        for i, data in enumerate(self.train_loader, 0):
            self.timer['iter time'].tic()
            if cfg.DATASET == 'SHTRGBD':
                img, gt_map, depth, ann, target_wh, reg_mask, ind, fname = data
                # print('ind, ann, reg_mask, target_wh', ind, ann, reg_mask, target_wh)
                img = Variable(img).cuda()
                gt_map = gt_map.float()
                gt_map = Variable(gt_map).cuda()
                depth = Variable(depth).cuda()
                ann = Variable(ann).cuda()
                target_wh = Variable(target_wh).cuda()
                reg_mask = Variable(reg_mask).cuda()
                ind = Variable(ind).cuda()
                # vis.heatmap(gt_map[0], win="heatmap", opts=dict(title="heatmap"))
            else:
                img, gt_map, fname  = data
                gt_map = gt_map.float()
                img = Variable(img).cuda()
                gt_map = Variable(gt_map).cuda()
                # oa_gt = Variable(oa_gt).cuda()
                if cfg.VISUALIZATION:
                    vis.heatmap(gt_map[0].flip([0]).squeeze(), win="gt_map", opts=dict(title="gt_map"))
                    # vis.heatmap(oa_gt[0].flip([0]).squeeze(), win="oa_gt", opts=dict(title="oa_gt"))
                    plt.imshow(np.array(self.restore_transform(img[0].detach().cpu())))
                    vis.matplot(plt, win="img", opts=dict(title="img"))

            self.optimizer.zero_grad()
            if cfg.DATASET == 'SHTRGBD':
                # pred_map, wh_out = self.net(img, gt_map, depth, ann, target_wh, reg_mask, ind)
                # dm_loss, bbox_loss = self.net.loss
                # loss = cfg_data.DMLOSS_WEIGHT*dm_loss+cfg_data.BBOXLOSS_WEIGHT*bbox_loss
                pred_map,_ = self.net(img, gt_map, depth, ann, target_wh, reg_mask, ind)
                dm_loss, r_loss, d_loss, bbox_loss, sim_loss = self.net.loss
                loss = self.cfg_data.DMLOSS_WEIGHT*dm_loss+self.cfg_data.DMLOSS_WEIGHT*r_loss+self.cfg_data.DMLOSS_WEIGHT*d_loss+self.cfg_data.BBOXLOSS_WEIGHT*bbox_loss+0.1*sim_loss
            else:
                # if self.epoch in [50, 100]:
                #     self.net = self.DRLnet(img, self.net, self.depth)
                pred_map= self.net(img, gt_map)
                if cfg.VISUALIZATION:
                    vis.heatmap(pred_map[0].squeeze().flip([0]), win="pred_map", opts=dict(title="pred_map"))
                    # oa_map = torch.zeros_like(oa_pred)
                    # oa_map[oa_pred>0.5] = 1
                    # oa_map[oa_pred<=0.5] = 0
                    # vis.heatmap(oa_map[0].squeeze().flip([0]), win="oa_pred", opts=dict(title="oa_pred"))
                dm_loss= self.net.loss
                dm_losses+=dm_loss.item()
                losses+=dm_loss.item()
                loss = self.net.loss
            for j in range(pred_map.shape[0]):
                pred_single = pred_map[j].data.cpu().numpy()
                gt_single = gt_map[j].data.cpu().numpy()
                pred_cnt = np.sum(pred_single)/100
                gt_cnt = np.sum(gt_single)/100

                single_mae = abs(pred_cnt-gt_cnt)
                # print("fname:{}, pred_cnt:{}, gt_cnt:{}, single_mae: {}".format(fname[j], pred_cnt, gt_cnt, single_mae))
                mae+=single_mae
            loss.backward()
            self.optimizer.step()
            # self.scheduler.step()
            if (i + 1) % cfg.PRINT_FREQ == 0:
                self.i_tb += 1
                self.writer.add_scalar('train_loss', loss.item(), self.i_tb)
                self.timer['iter time'].toc(average=False)
                
                if cfg.DATASET == 'SHTRGBD':
                    self.file.write("[epoch:{}, iter:{}, dm_loss:{:.4f}, r_loss:{:.4f}, d_loss:{:.4f}, bbox_loss:{:.4f}, sim_loss:{:.4f}, total_loss:{:.4f}, lr:{:.8f}, iter_time:{:.2f}s]\n"\
                            .format(self.epoch+1, i+1, dm_loss.item(), r_loss.item(), d_loss.item(), bbox_loss.item(), sim_loss.item(), loss.item(), self.optimizer.param_groups[0]['lr']*10000, self.timer['iter time'].diff))
                    self.file.write('[cnt: gt: %.1f pred: %.2f]\n' % (gt_map[0].sum().data / self.cfg_data.LOG_PARA, pred_map[0][0].sum().data / self.cfg_data.LOG_PARA))
                else:
                    # file.write("[epoch:{}, iter:{}, loss:{:.4f}, lr:{:.8f}, iter_time:{:.2f}s]\n"\
                    #         .format(self.epoch+1, i+1, loss.item(), self.optimizer.param_groups[0]['lr']*10000, self.timer['iter time'].diff))
                    # file.write('[cnt: gt: %.1f pred: %.2f]\n' % (gt_map[0].sum().data / self.cfg_data.LOG_PARA, pred_map[0][0].sum().data / self.cfg_data.LOG_PARA))
                    self.file.write("[epoch:{}, iter:{}, dm_loss:{:.4f}, total_loss:{:.4f}, lr:{:.8f}, iter_time:{:.2f}s]\n"\
                            .format(self.epoch+1, i+1, dm_loss.item(), loss.item(), self.optimizer.param_groups[0]['lr']*10000, self.timer['iter time'].diff))
                    self.file.write('[cnt: gt: %.1f pred: %.2f]\n' % (gt_map[0].sum().data / self.cfg_data.LOG_PARA, pred_map[0][0].sum().data / self.cfg_data.LOG_PARA))
                if cfg.DATASET == 'SHTRGBD':
                    print("[epoch:{}, iter:{}, dm_loss:{:.4f}, r_loss:{:.4f}, d_loss:{:.4f}, bbox_loss:{:.4f}, sim_loss:{:.4f}, total_loss:{:.4f}, lr:{:.8f}, iter_time:{:.2f}s]" \
                          .format(self.epoch+1, i+1, dm_loss.item(), r_loss.item(), d_loss.item(), bbox_loss.item(), sim_loss.item(), loss.item(), self.optimizer.param_groups[0]['lr']*10000, self.timer['iter time'].diff))
                    print('[cnt: gt: %.1f pred: %.2f]' % (gt_map[0].sum().data / self.cfg_data.LOG_PARA, pred_map[0][0].sum().data / self.cfg_data.LOG_PARA))
                else:
                    # print("[epoch:{}, iter:{}, loss:{:.4f}, lr:{:.8f}, iter_time:{:.2f}s]" \
                    #       .format(self.epoch+1, i+1, loss.item(), self.optimizer.param_groups[0]['lr']*10000, self.timer['iter time'].diff))
                    # print('[cnt: gt: %.1f pred: %.2f]' % (gt_map[0].sum().data / self.cfg_data.LOG_PARA, pred_map[0][0].sum().data / self.cfg_data.LOG_PARA))    
                    print("[epoch:{}, iter:{}, dm_loss:{:.4f}, total_loss:{:.4f}, lr:{:.8f}, iter_time:{:.2f}s]" \
                          .format(self.epoch+1, i+1, dm_loss.item(), loss.item(), self.optimizer.param_groups[0]['lr']*10000, self.timer['iter time'].diff))
                    print('[cnt: gt: %.1f pred: %.2f]' % (gt_map[0].sum().data / self.cfg_data.LOG_PARA, pred_map[0][0].sum().data / self.cfg_data.LOG_PARA))
        mae/=(len(self.train_loader)*self.cfg_data.TRAIN_BATCH_SIZE)
        vis.line([[losses/len(self.train_loader)]], [self.epoch], win='train_loss', update='append')
        vis.line([[dm_losses/len(self.train_loader)]], [self.epoch], win='dm_loss', update='append')
        print("-------------------------------------[Epoch:{}, Train_MAE:{:.2f}]-------------------------------------".format(self.epoch, mae))       


    def validate_V1(self):# validate_V1 for SHHA, SHHB, UCF-QNRF, UCF50

        self.net.eval()
        # self.net.load_state_dict(torch.load('./epoch0.pth'))
        losses = AverageMeter()
        maes = AverageMeter()
        mses = AverageMeter()
        PSNR, SSIM = AverageMeter(), AverageMeter()
        for vi, data in enumerate(tqdm(self.val_loader), 0):
            if cfg.DATASET == 'SHTRGBD':
                img, gt_map, depth, ann, target_wh, reg_mask, ind, fname = data
            else:
                img, gt_map, fname = data
            # print("validation iter:{}".format(vi))
            with torch.no_grad():
                img = Variable(img).cuda()
                gt_map = Variable(gt_map).cuda()
                # oa_gt = Variable(oa_gt).cuda()
                if cfg.DATASET == 'SHTRGBD':
                    depth = Variable(depth).cuda()
                    ind = Variable(ind).cuda()
                    target_wh = Variable(target_wh).cuda()
                    reg_mask = Variable(reg_mask).cuda()
                # if cfg.VISUALIZATION:
                #     vis.heatmap(gt_map[0].flip([0]).squeeze(), win="gt_map", opts=dict(title="gt_map"))
                #     vis.heatmap(oa_gt[0].flip([0]).squeeze(), win="oa_gt", opts=dict(title="oa_gt"))
                #     plt.imshow(np.array(self.restore_transform(img[0].detach().cpu())))
                #     vis.matplot(plt, win="img", opts=dict(title="img"))

                if cfg.DATASET == 'SHTRGBD':
                    pred_map = self.net.forward(img, gt_map, depth, ann, target_wh, reg_mask, ind=None)
                else:
                    pred_map = self.net.forward(img, gt_map)
                if cfg.VISUALIZATION:
                    vis.heatmap(pred_map[0].squeeze().flip([0]), win="pred_map", opts=dict(title="pred_map"))
                    # oa_map = torch.zeros_like(oa_pred)
                    # oa_map[oa_pred>0.5] = 1
                    # oa_map[oa_pred<=0.5] = 0
                    vis.heatmap(gt_map[0].squeeze().flip([0]), win="gt_map", opts=dict(title="gt_map"))
                # ind = ind.data.cpu().numpy()
                # target_wh = target_wh.data.cpu().numpy()
                # reg_mask = reg_mask.data.cpu().numpy()
                # depth = depth.data.cpu().numpy()

                pred_map = pred_map.data.cpu().numpy()
                gt_map = gt_map.data.cpu().numpy()

                for i_img in range(pred_map.shape[0]):
                    # single_PSNR, single_SSIM = evaluate(pred_map[i_img].squeeze()/self.cfg_data.LOG_PARA, gt_map[i_img].squeeze()/self.cfg_data.LOG_PARA)
                    # PSNR.update(single_PSNR)
                    # SSIM.update(single_SSIM)
                    pred_cnt = np.sum(pred_map[i_img])/self.cfg_data.LOG_PARA
                    gt_count = np.sum(gt_map[i_img])/self.cfg_data.LOG_PARA

                    if isinstance(self.net.loss, tuple):
                        losses.update(self.net.loss[0].item()+self.net.loss[1].item())
                    else:
                        losses.update(self.net.loss.item())
                    single_mae = abs(gt_count-pred_cnt)
                    single_mse = (gt_count-pred_cnt)*(gt_count-pred_cnt)
                    # self.save_feature(gt_map.squeeze(), dir='./visout', fname=fname[0].split('.')[0], gt='_gt{:.2f}'.format(gt_count))
                    # self.save_feature(pred_map.squeeze(), dir='./visout', fname=fname[0].split('.')[0], pred='_pred{:.2f}'.format(pred_cnt))

                    maes.update(abs(gt_count-pred_cnt))
                    mses.update((gt_count-pred_cnt)*(gt_count-pred_cnt))
                if vi==0:
                    vis_results(self.exp_name, self.epoch, self.writer, self.restore_transform, img, pred_map, gt_map)
            
        mae = maes.avg
        mse = np.sqrt(mses.avg)
        loss = losses.avg
        psnr, ssim = PSNR.avg, SSIM.avg
        self.writer.add_scalar('val_loss', loss, self.epoch + 1)
        self.writer.add_scalar('mae', mae, self.epoch + 1)
        self.writer.add_scalar('mse', mse, self.epoch + 1)

        self.train_record = update_model(self.net,self.optimizer,self.scheduler,self.epoch,self.i_tb,self.exp_path,self.exp_name, \
            [mae, mse, loss],self.train_record,self.log_txt)
        print_summary(self.exp_name,[mae, mse, loss],self.train_record)


    def validate_V2(self):# validate_V2 for WE
        small = [11, 15, 16, 22, 23, 24, 34, 35, 42, 43, 44, 61, 62, 63, 65, 69, 70, 74]
        large = [17, 18, 75, 82, 88, 95, 101, 103, 105, 108, 111, 112]
        cloudy = [11, 22, 23, 24, 43, 61, 62, 63, 65, 75, 82, 88, 95, 101, 108]
        sunny = [15, 16, 34, 35, 42, 44, 103, 105, 111, 112]  
        night = [17, 18, 69, 70, 74]
        crowd = [11, 15, 22, 23, 24, 88, 101, 105, 108, 111, 112]
        sparse = [16, 17, 18, 34, 35, 42, 43, 44, 61, 62, 63, 65, 69, 70, 74, 75, 82, 95, 103]
        mae_small, mae_large, mae_cloudy, mae_sunny, mae_night, mae_crowd, mae_sparse = AverageMeter(),AverageMeter(),AverageMeter(),AverageMeter(),AverageMeter(),AverageMeter(),AverageMeter()
        mse_small, mse_large, mse_cloudy, mse_sunny, mse_night, mse_crowd, mse_sparse = AverageMeter(),AverageMeter(),AverageMeter(),AverageMeter(),AverageMeter(),AverageMeter(),AverageMeter()
        count_small, count_large, count_cloudy, count_sunny, count_night, count_crowd, count_sparse = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
        self.net.eval()
        # self.net.load_state_dict(torch.load('./epoch0.pth'))
        losses = AverageMeter()
        maes = AverageMeter()
        mses = AverageMeter()
        T = AverageMeter()
        # PSNR, SSIM = AverageMeter(), AverageMeter()
        for vi, data in enumerate(tqdm(self.val_loader), 0):
            img, gt_map, oa_gt, fname = data
            # print("validation iter:{}".format(vi))
            with torch.no_grad():
                img = Variable(img).cuda()
                gt_map = Variable(gt_map).cuda()
                oa_gt = Variable(oa_gt).cuda()
                if cfg.DATASET == 'SHTRGBD':
                    depth = Variable(depth).cuda()
                    ind = Variable(ind).cuda()
                    target_wh = Variable(target_wh).cuda()
                    reg_mask = Variable(reg_mask).cuda()
                # if cfg.VISUALIZATION:
                #     vis.heatmap(gt_map[0].flip([0]).squeeze(), win="gt_map", opts=dict(title="gt_map"))
                #     vis.heatmap(oa_gt[0].flip([0]).squeeze(), win="oa_gt", opts=dict(title="oa_gt"))
                #     plt.imshow(np.array(self.restore_transform(img[0].detach().cpu())))
                #     vis.matplot(plt, win="img", opts=dict(title="img"))
                    
                t1=time.time()
                pred_map, oa_pred = self.net.forward(img, gt_map, oa_gt)
                t2=time.time()
                T.update(1/(t2-t1))

                if cfg.VISUALIZATION:
                    vis.heatmap(pred_map[0].squeeze().flip([0]), win="pred_map", opts=dict(title="pred_map"))
                    # oa_map = torch.zeros_like(oa_pred)
                    # oa_map[oa_pred>0.5] = 1
                    # oa_map[oa_pred<=0.5] = 0
                    vis.heatmap(gt_map[0].squeeze().flip([0]), win="gt_map", opts=dict(title="gt_map"))
                # ind = ind.data.cpu().numpy()
                # target_wh = target_wh.data.cpu().numpy()
                # reg_mask = reg_mask.data.cpu().numpy()
                # depth = depth.data.cpu().numpy()

                pred_map = pred_map.data.cpu().numpy()
                gt_map = gt_map.data.cpu().numpy()

                for i_img in range(pred_map.shape[0]):
                    single_img_scene = int(fname[i_img].split('.')[0][3:6])
                    # single_PSNR, single_SSIM = evaluate(pred_map[i_img].squeeze()/self.cfg_data.LOG_PARA, gt_map[i_img].squeeze()/self.cfg_data.LOG_PARA)
                    # PSNR.update(single_PSNR)
                    # SSIM.update(single_SSIM)

                    pred_cnt = np.sum(pred_map[i_img])/self.cfg_data.LOG_PARA
                    gt_count = np.sum(gt_map[i_img])/self.cfg_data.LOG_PARA

                    if isinstance(self.net.loss, tuple):
                        losses.update(self.net.loss[0].item()+self.net.loss[1].item())
                    else:
                        losses.update(self.net.loss.item())
                    single_mae = abs(gt_count-pred_cnt)
                    single_mse = (gt_count-pred_cnt)*(gt_count-pred_cnt)
                    if single_img_scene in small:
                        mae_small.update(single_mae)
                        mse_small.update(single_mse)
                        # count_small.update(1
                    else:
                        mae_large.update(single_mae)
                        mse_large.update(single_mse)
                        # count_large.update(1

                    if single_img_scene in crowd:
                        mae_crowd.update(single_mae)
                        mse_crowd.update(single_mse)
                        # count_crowd.update(1
                    else:
                        mae_sparse.update(single_mae)
                        mse_sparse.update(single_mse)
                        # count_sparse.update(1
                
                    if single_img_scene in cloudy:
                        mae_cloudy.update(single_mae)
                        mse_cloudy.update(single_mse)
                        # count_cloudy.update(1
                    elif single_img_scene in sunny:
                        mae_sunny.update(single_mae)
                        mse_sunny.update(single_mse)
                        # count_sunny.update(1
                    else:
                        mae_night.update(single_mae)
                        mse_night.update(single_mse)
                        # count_night += 1
                    # self.save_feature(gt_map.squeeze(), dir='./visout', fname=fname[0].split('.')[0], gt='_gt{:.2f}'.format(gt_count))
                    # self.save_feature(pred_map.squeeze(), dir='./visout', fname=fname[0].split('.')[0], pred='_pred{:.2f}'.format(pred_cnt))
                    
                    maes.update(abs(gt_count-pred_cnt))
                    mses.update((gt_count-pred_cnt)*(gt_count-pred_cnt))
                if vi==0:
                    vis_results(self.exp_name, self.epoch, self.writer, self.restore_transform, img, pred_map, gt_map)
            
        mae = maes.avg
        mse = np.sqrt(mses.avg)
        loss = losses.avg
        fps = T.avg
        mae_small, mae_large, mae_cloudy, mae_sunny, mae_night, mae_crowd, mae_sparse = mae_small.avg, mae_large.avg, mae_cloudy.avg, mae_sunny.avg, mae_night.avg, mae_crowd.avg, mae_sparse.avg
        mse_small, mse_large, mse_cloudy, mse_sunny, mse_night, mse_crowd, mse_sparse = np.sqrt(mse_small.avg), np.sqrt(mse_large.avg), np.sqrt(mse_cloudy.avg), np.sqrt(mse_sunny.avg), np.sqrt(mse_night.avg), np.sqrt(mse_crowd.avg), np.sqrt(mse_sparse.avg)
        print(f'{mae_small},{mae_large}, {mae_cloudy}, {mae_sunny}, {mae_night}, {mae_crowd}, {mae_sparse}')
        print(f'{mse_small}, {mse_large}, {mse_cloudy}, {mse_sunny}, {mse_night}, {mse_crowd}, {mse_sparse}')
        # psnr, ssim = PSNR.avg, SSIM.avg
        self.writer.add_scalar('val_loss', loss, self.epoch + 1)
        self.writer.add_scalar('mae', mae, self.epoch + 1)
        self.writer.add_scalar('mse', mse, self.epoch + 1)

        self.train_record = update_model(self.net,self.optimizer,self.scheduler,self.epoch,self.i_tb,self.exp_path,self.exp_name, \
            [mae, mse, loss],self.train_record,self.log_txt)
        print_summary(self.exp_name,[mae, mse, loss],self.train_record)





    def validate_V3(self):# validate_V3 for GCC

        self.net.eval()
        
        losses = AverageMeter()
        maes = AverageMeter()
        mses = AverageMeter()

        c_maes = {'level':AverageCategoryMeter(9), 'time':AverageCategoryMeter(8),'weather':AverageCategoryMeter(7)}
        c_mses = {'level':AverageCategoryMeter(9), 'time':AverageCategoryMeter(8),'weather':AverageCategoryMeter(7)}


        for vi, data in enumerate(self.val_loader, 0):
            img, gt_map, attributes_pt = data

            with torch.no_grad():
                img = Variable(img).cuda()
                gt_map = Variable(gt_map).cuda()


                pred_map = self.net.forward(img,gt_map)

                pred_map = pred_map.data.cpu().numpy()
                gt_map = gt_map.data.cpu().numpy()

                for i_img in range(pred_map.shape[0]):
                
                    pred_cnt = np.sum(pred_map[i_img])/self.cfg_data.LOG_PARA
                    gt_count = np.sum(gt_map[i_img])/self.cfg_data.LOG_PARA

                    s_mae = abs(gt_count-pred_cnt)
                    s_mse = (gt_count-pred_cnt)*(gt_count-pred_cnt)

                    losses.update(self.net.loss.item())
                    maes.update(s_mae)
                    mses.update(s_mse)   
                    attributes_pt = attributes_pt.squeeze() 
                    c_maes['level'].update(s_mae,attributes_pt[i_img][0])
                    c_mses['level'].update(s_mse,attributes_pt[i_img][0])
                    c_maes['time'].update(s_mae,attributes_pt[i_img][1]/3)
                    c_mses['time'].update(s_mse,attributes_pt[i_img][1]/3)
                    c_maes['weather'].update(s_mae,attributes_pt[i_img][2])
                    c_mses['weather'].update(s_mse,attributes_pt[i_img][2])


                if vi==0:
                    vis_results(self.exp_name, self.epoch, self.writer, self.restore_transform, img, pred_map, gt_map)
            
        loss = losses.avg
        mae = maes.avg
        mse = np.sqrt(mses.avg)


        self.writer.add_scalar('val_loss', loss, self.epoch + 1)
        self.writer.add_scalar('mae', mae, self.epoch + 1)
        self.writer.add_scalar('mse', mse, self.epoch + 1)

        self.train_record = update_model(self.net,self.optimizer,self.scheduler,self.epoch,self.i_tb,self.exp_path,self.exp_name, \
            [mae, mse, loss],self.train_record,self.log_txt)


        print_GCC_summary(self.log_txt,self.epoch,[mae, mse, loss],self.train_record,c_maes,c_mses)

    def split_parameters(self, module):
        params_decay = []
        params_no_decay = []
        # params_bias = []
        # for k, v in module.named_modules():
        #     if hasattr(v, 'bias') and isinstance(v.bias, nn.parameter):
        #         params_bias.append(v.bias)
        #     if isinstance(v, nn.BatchNorm2d):
        #         params_no_decay.append(v.weight)
        #     elif hasattr(v, 'weight') and isinstance(v.weight, nn.parameter):
        #         params_decay.append(v.weight)
        for m in module.modules():
            if isinstance(m, torch.nn.Linear):
                params_decay.append(m.weight)
                if m.bias is not None:
                    params_no_decay.append(m.bias)
            elif isinstance(m, torch.nn.modules.conv._ConvNd):
                params_decay.append(m.weight)
                if m.bias is not None:
                    params_no_decay.append(m.bias)
            elif isinstance(m, torch.nn.modules.batchnorm._BatchNorm):
                params_no_decay.extend([*m.parameters()])
            elif len(list(m.children())) == 0:
                params_decay.extend([*m.parameters()])
        assert len(list(module.parameters())) == len(params_decay) + len(params_no_decay)
        return params_decay, params_no_decay

    def validate_V4(self):# validate_V1 for SHHA, SHHB, UCF-QNRF, UCF50

        self.net.eval()
        # self.net.load_state_dict(torch.load('./epoch0.pth'))
        losses = AverageMeter()
        maes = AverageMeter()
        mses = AverageMeter()

        for vi, data in enumerate(tqdm(self.val_loader), 0):
            if cfg.DATASET == 'SHTRGBD':
                img, gt_map, depth, ann, target_wh, reg_mask, ind, fname = data
            else:
                img, gt_map, oa_gt, fname = data
            # print("validation iter:{}".format(vi))
            with torch.no_grad():
                img = Variable(img).cuda()
                gt_map = Variable(gt_map).cuda()
                oa_gt = Variable(oa_gt).cuda()
                b, c, h, w = img.shape
                pad_size_h = h//self.cfg_data.STD_SIZE[0]+1
                pad_size_w = w//self.cfg_data.STD_SIZE[1]+1
                pad_h = pad_size_h*self.cfg_data.STD_SIZE[0]-h
                pad_w = pad_size_w*self.cfg_data.STD_SIZE[1]-w

                img = F.pad(img, pad=(0, pad_w, 0, pad_h),mode='constant', value=0.0)
                gt_map = F.pad(gt_map, pad=(0, pad_w, 0, pad_h),mode='constant', value=0.0)
                oa_gt = F.pad(oa_gt, pad=(0, pad_w, 0, pad_h),mode='constant', value=0.0)


                pred_oa = torch.zeros_like(oa_gt).cpu()
                batch_cnt = 0.0
                batch_loss = 0.0
                for i in range(pad_size_h):
                    for j in range(pad_size_w):
                        pad_img = img[:,:, i*self.cfg_data.STD_SIZE[0]:(i+1)*self.cfg_data.STD_SIZE[0], j*self.cfg_data.STD_SIZE[1]:(j+1)*self.cfg_data.STD_SIZE[1]]
                        pad_gt_map = gt_map[:, i*self.cfg_data.STD_SIZE[0]:(i+1)*self.cfg_data.STD_SIZE[0], j*self.cfg_data.STD_SIZE[1]:(j+1)*self.cfg_data.STD_SIZE[1]]
                        pad_oa_gt = oa_gt[:, i*self.cfg_data.STD_SIZE[0]:(i+1)*self.cfg_data.STD_SIZE[0], j*self.cfg_data.STD_SIZE[1]:(j+1)*self.cfg_data.STD_SIZE[1]]
                        pred_map, oa_pred = self.net.forward(pad_img, pad_gt_map, pad_oa_gt)
                        pred_oa[:, i*self.cfg_data.STD_SIZE[0]:(i+1)*self.cfg_data.STD_SIZE[0], j*self.cfg_data.STD_SIZE[1]:(j+1)*self.cfg_data.STD_SIZE[1]] = pred_map.squeeze().detach().cpu()
                        # if cfg.VISUALIZATION:
                        #     vis.heatmap(pad_gt_map[0].flip([0]).squeeze(), win="gt_map", opts=dict(title="gt_map"))
                        #     vis.heatmap(pad_oa_gt[0].flip([0]).squeeze(), win="oa_gt", opts=dict(title="oa_gt"))
                        #     plt.imshow(np.array(self.restore_transform(pad_img[0].detach().cpu())))
                        #     vis.matplot(plt, win="img", opts=dict(title="img"))  

                        pred_map = pred_map.data.cpu().numpy()
                        # pad_gt_map = pad_gt_map.data.cpu().numpy()
                        chunk_cnt_pred = np.sum(pred_map.squeeze())
                        # chunk_cnt_gt = np.sum(pad_gt_map.squeeze())
                        batch_cnt+=chunk_cnt_pred
                        batch_loss+=(self.net.loss[0].item()+self.net.loss[0].item())
                        j+=1
                    i+=1  
                if cfg.VISUALIZATION:
                    vis.heatmap(pad_gt_map.squeeze().flip([0]), win="pred_map", opts=dict(title="pred_map"))
                for i_img in range(pred_map.shape[0]):
                    gt_map = gt_map.data.cpu().numpy()
                    pred_cnt = batch_cnt/self.cfg_data.LOG_PARA
                    gt_count = np.sum(gt_map[i_img])/self.cfg_data.LOG_PARA
                    # self.save_feature(gt_map[:,0:h, 0:w].squeeze(), dir='./visout', fname=fname[0].split('.')[0], gt='_gt{:.2f}'.format(gt_count))
                    # self.save_feature(pred_oa[0,0:h, 0:w].squeeze().numpy(), dir='./visout', fname=fname[0].split('.')[0], pred='_pred{:.2f}'.format(pred_cnt))
                    if fname[0]=='P0706.png':
                        mdict = {}
                        mdict[f'gt_map'] = gt_map[:,0:h, 0:w].squeeze()
                        mdict[f'pred_map'] = pred_oa[0,0:h, 0:w].squeeze().numpy()
                        savemat("./figure1_pred.mat".format(fname), mdict)
                    maes.update(abs(gt_count-pred_cnt))
                    mses.update((gt_count-pred_cnt)*(gt_count-pred_cnt))
                    losses.update(batch_loss)
        mae = maes.avg
        mse = np.sqrt(mses.avg)
        loss = losses.avg

        self.writer.add_scalar('val_loss', loss, self.epoch + 1)
        self.writer.add_scalar('mae', mae, self.epoch + 1)
        self.writer.add_scalar('mse', mse, self.epoch + 1)

        self.train_record = update_model(self.net,self.optimizer,self.scheduler,self.epoch,self.i_tb,self.exp_path,self.exp_name, \
            [mae, mse, loss],self.train_record,self.log_txt)
        print_summary(self.exp_name,[mae, mse, loss],self.train_record)
        return mae


    def validate_V5(self):# validate_V1 for SHHA, SHHB, UCF-QNRF, UCF50

        self.net.eval()
        # self.net.load_state_dict(torch.load('./epoch0.pth'))
        losses = AverageMeter()
        maes = AverageMeter()
        mses = AverageMeter()

        for vi, data in enumerate(tqdm(self.val_loader), 0):
            if cfg.DATASET == 'SHTRGBD':
                img, gt_map, depth, ann, target_wh, reg_mask, ind, fname = data
            else:
                img, gt_map, oa_gt, fname = data
            # print("validation iter:{}".format(vi))
            with torch.no_grad():
                # img = Variable(img).cuda()
                # gt_map = Variable(gt_map).cuda()
                # oa_gt = Variable(oa_gt).cuda()
                b, c, h, w = img.shape
                pad_size_h = h//self.cfg_data.STD_SIZE[0]+1
                pad_size_w = w//self.cfg_data.STD_SIZE[1]+1
                pad_h = pad_size_h*self.cfg_data.STD_SIZE[0]-h
                pad_w = pad_size_w*self.cfg_data.STD_SIZE[1]-w

                img = F.pad(img, pad=(0, pad_w, 0, pad_h),mode='constant', value=0.0)
                gt_map = F.pad(gt_map, pad=(0, pad_w, 0, pad_h),mode='constant', value=0.0)
                oa_gt = F.pad(oa_gt, pad=(0, pad_w, 0, pad_h),mode='constant', value=0.0)

                batch_cnt = 0.0
                batch_loss = 0.0
                for i in trange(pad_size_h):
                    for j in range(pad_size_w):
                        pad_img = img[:,:, i*self.cfg_data.STD_SIZE[0]:(i+1)*self.cfg_data.STD_SIZE[0], j*self.cfg_data.STD_SIZE[1]:(j+1)*self.cfg_data.STD_SIZE[1]]
                        pad_gt_map = gt_map[:, i*self.cfg_data.STD_SIZE[0]:(i+1)*self.cfg_data.STD_SIZE[0], j*self.cfg_data.STD_SIZE[1]:(j+1)*self.cfg_data.STD_SIZE[1]]
                        pad_oa_gt = oa_gt[:, i*self.cfg_data.STD_SIZE[0]:(i+1)*self.cfg_data.STD_SIZE[0], j*self.cfg_data.STD_SIZE[1]:(j+1)*self.cfg_data.STD_SIZE[1]]
                        pad_img = Variable(pad_img).cuda()
                        pad_gt_map = Variable(pad_gt_map).cuda()
                        pad_oa_gt = Variable(pad_oa_gt).cuda()
                        pred_map, oa_pred = self.net.forward(pad_img, pad_gt_map, pad_oa_gt)

                        if cfg.VISUALIZATION:
                            vis.heatmap(pad_gt_map[0].flip([0]).squeeze(), win="gt_map", opts=dict(title="gt_map"))
                            vis.heatmap(pad_oa_gt[0].flip([0]).squeeze(), win="oa_gt", opts=dict(title="oa_gt"))
                            plt.imshow(np.array(self.restore_transform(pad_img[0].detach().cpu())))
                            vis.matplot(plt, win="img", opts=dict(title="img"))  

                        pred_map = pred_map.data.cpu().numpy()
                        # pad_gt_map = pad_gt_map.data.cpu().numpy()
                        chunk_cnt_pred = np.sum(pred_map.squeeze())
                        # chunk_cnt_gt = np.sum(pad_gt_map.squeeze())
                        batch_cnt+=chunk_cnt_pred
                        batch_loss+=(self.net.loss[0].item()+self.net.loss[0].item())
                        j+=1
                    i+=1            
                for i_img in range(pred_map.shape[0]):
                    gt_map = gt_map.data.cpu().numpy()
                    pred_cnt = batch_cnt/self.cfg_data.LOG_PARA
                    gt_count = np.sum(gt_map[i_img])/self.cfg_data.LOG_PARA
                    maes.update(abs(gt_count-pred_cnt))
                    mses.update((gt_count-pred_cnt)*(gt_count-pred_cnt))
                    losses.update(batch_loss)
        mae = maes.avg
        mse = np.sqrt(mses.avg)
        loss = losses.avg

        self.writer.add_scalar('val_loss', loss, self.epoch + 1)
        self.writer.add_scalar('mae', mae, self.epoch + 1)
        self.writer.add_scalar('mse', mse, self.epoch + 1)

        self.train_record = update_model(self.net,self.optimizer,self.scheduler,self.epoch,self.i_tb,self.exp_path,self.exp_name, \
            [mae, mse, loss],self.train_record,self.log_txt)
        print_summary(self.exp_name,[mae, mse, loss],self.train_record)

    def save_feature(self, feature_map_to_save, fname, dir = None, pred=None, gt=None, ext=None):
        feature_map_to_save = (feature_map_to_save - np.min(feature_map_to_save)) / np.ptp(feature_map_to_save) * 255
        feature_map_colored = cv2.applyColorMap(np.uint8(feature_map_to_save), cv2.COLORMAP_JET)
        if pred == None:
            pred = ''
        if gt==None:
            gt=''
        if ext==None:
            ext=''
        if not os.path.exists(dir):
            os.mkdir(dir)
        cv2.imwrite(f'{dir}/{fname}{pred}{gt}{ext}.png', feature_map_colored)


def normalize_feature(feature):
    # 根据特征的范围进行归一化，例如，将其缩放到 [0, 1]
    normalized_feature = (feature - feature.min()) / (feature.max() - feature.min())
    return normalized_feature

def calculate_psnr(feature1, feature2):
    mse = np.mean((feature1 - feature2) ** 2)
    if mse == 0:
        return float('inf')
    max_pixel = 1.0  # 如果特征被归一化到 [0, 1]，否则根据实际情况设置
    psnr = 10 * np.log10((max_pixel ** 2) / mse)
    return psnr

def calculate_ssim(feature1, feature2):
    # Convert features to 8-bit (0-255) for SSIM calculation
    feature1_8bit = (feature1 * 255).astype(np.uint8)
    feature2_8bit = (feature2 * 255).astype(np.uint8)

    # Calculate SSIM
    ssim_value, _ = ssim(feature1_8bit, feature2_8bit, full=True)
    return ssim_value

def evaluate(feature1, feature2):
    # 假设 feature1 和 feature2 是两个单通道特征的 NumPy 数组
    # feature1 = np.random.rand(100, 100)  # 临时随机生成示例特征
    # feature2 = np.random.rand(100, 100)  # 临时随机生成示例特征

    # 归一化特征
    normalized_feature1 = normalize_feature(feature1)
    normalized_feature2 = normalize_feature(feature2)

    # 计算PSNR和SSIM
    psnr_value = calculate_psnr(normalized_feature1, normalized_feature2)
    ssim_value = calculate_ssim(normalized_feature1, normalized_feature2)

    print(f"PSNR: {psnr_value} dB")
    print(f"SSIM: {ssim_value}")
    return psnr_value, ssim_value