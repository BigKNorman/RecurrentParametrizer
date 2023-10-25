import argparse 
import os
import random
import math

import matplotlib
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributions as dist
from torch.utils.data import Dataset, DataLoader

from torchdiffeq import odeint

import numpy as np

from utils.utility import save_state, load_net_state, load_seed, load_opt_state
from utils.lin_interp_utils import get_norm_mags, get_targets 
from utils.utility_s2d import normalize, unnormalize, b_normalize, b_unnormalize
#from utils.utility_c3d import 3dnormalize, 3dunnormalize, 3db_normalize, 3db_unnormalize #write these functions? probs nbd 
from utils.p2v2dutils import init, voxelize_pointcloud, reconstruct_2d_pc 
from utils.p2v3dutils import init3d, voxelize_pointcloud3d, reconstruct_3d_pc #TODO return these to 3d pc implementations

from Losses.msetrajloss import TrajectorySELoss
from Losses.autonomousmseloss import AutonomousMSELoss

from data_loader_ode import Simple2dDatasetMagLVCUnenc as S2DDataset
from data_loader_ur5 import UR5CubbyDataset
from data_loader_ur5 import UR5CubbyValDataset 
from data_loader_ur5 import UR5CubbyStepwiseDataset 
from data_loader_ur5 import UR5CubbyStepwiseValDataset 
from data_loader_ode import load_test_dataset, load_dataset

from Models.ODEFuncs.mpnetlikefb import MPNetLikeFeedback
from Models.ODEBlocks.latent_block import LatentBlock
from Models.ODENets.latent_odenet import LatentODENet
from Models.NeuralFlows.flow import ResNetFlow
from Models.NeuralFlows.flow import CouplingFlow
from Models.FFPlanner.ff import FF

from Models.EnvNets.ff import FF as FFEnc
from Models.EnvNets.cnn2d import CNN
from Models.EnvNets.cnn2dwch import CNNHCH
from Models.EnvNets.cnn3d import CNN as CNN3d
from Models.EnvNets.pointnet2_msg import PointNet2
from Models.EnvNets.PointNetFromPaper.pointnet import PointNet
from Models.EnvNets.ndfnet2d import NDF
from Models.EnvNets.ndfnet3d import NDF as NDF3d

list_of_512_enc = ["pnet", "pnet++", "ff"]
list_of_vox_enc = ["cnn", "ndf", "cnnhch"]
list_of_rec_enc = ["pnet", "pnet++"]
list_of_sg_enc = ["pnet", "pnet++"]
list_of_env_enc = ["pnet", "pnet++"]
list_of_cont_planners = ["rf", "cf"]

#["pnet", "pnet++", "ff"]


#for saving and loading model
class NetWrapper(torch.nn.Module): 
    def __init__(self, ndf, rnf, st_enc, st_dec, pos_dec, opt): 
        super().__init__()
        self.ndf = ndf
        self.rnf = rnf
        self.st_enc = st_enc
        self.st_dec = st_dec     
        self.pos_dec = pos_dec 
        self.opt = opt 



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str,
                        default='./models/', help='path for saving trained models')
    parser.add_argument('--no_env', type=int, default=50,
                        help='directory for obstacle images, if ur5 env then only 1 training env') #was 50, that's the goal
    parser.add_argument('--num-epochs', type=int, default=100000,
                        help='number of epochs to train to') 

    parser.add_argument('--no_motion_paths', type=int, default=2000,
                        help='number of optimal paths in each environment, if ur5 env then only 4000 training env')
    parser.add_argument('--log_step', type=int, default=10,
                        help='step size for prining log info')
    parser.add_argument('--save_step', type=int, default=10,
                        help='step size for saving trained models')
    parser.add_argument('--save-epoch', type=int, default=1000)

    # Model parameters
    #parser.add_argument('--output-size', type=int, default=2,
    #                    help='dimension of the input vector')
    parser.add_argument('--hidden-size', type=int, default=256,
                        help='dimension of lstm hidden states')
    parser.add_argument('--n-layers', type=int, default=7, 
                        help='dimension of lstm hidden states')

    parser.add_argument('--env-type', type=str, default='simple2d', help='which environment dataset to train with: simple2d, complex3d, pianomover, ur5cubby')
    parser.add_argument('--encoder-type', type=str, default='ff', help='encoder to use for model: ff, cnn, pnet, pnet++, ndf')
    parser.add_argument('--planner-type', type=str, default='rf', help='planner type to use: rf, cf, ff')

    parser.add_argument('--start-epoch', type=int, default=0,
                        help='epoch to start training at')
    parser.add_argument('--model-path', type=str, default='./models',
                        help='epoch to start training at')


    parser.add_argument('--batch-size', type=int, default=128) #was 64 for mseloss
    parser.add_argument('--learning-rate', type=float, default=1e-4)
    parser.add_argument('--time-samples', type=int, default=10) #was 50, though I will guess... I don't think it matters 
    #parser.add_argument('--world-size', type=int, default=20)
    parser.add_argument('--solver-tolerance', type=float, default=1e-3)
    parser.add_argument('--pc-samps', type=float, default=0) #wanna do 2048?
    args = parser.parse_args()
    print(args)

    torch_seed = np.random.randint(low=0, high=1000)
    np_seed = np.random.randint(low=0, high=1000)
    py_seed = np.random.randint(low=0, high=1000)

    torch.manual_seed(torch_seed)
    np.random.seed(np_seed)
    random.seed(py_seed)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'


    batch_size = args.batch_size
    hidden_dim = args.hidden_size
    dim = hidden_dim
    n_layers = args.n_layers
    if args.encoder_type == "cnn": 
        env_enc_dim = 2704 #this is wrong... but idk what this should be?
    elif args.encoder_type == "cnnhch": 
        env_enc_dim = 3136 #this is wrong... but idk what this should be?
    elif args.encoder_type == "ndf": 
        env_enc_dim = 2487 #this is wrong... but idk what this should be?
    elif args.encoder_type in list_of_512_enc:  
        env_enc_dim = 512 #this is wrong... but idk what this should be?


    if args.env_type == "simple2d": 
        output_dim = 2
        space_dim = 2
        pcdimunflat = [1400, space_dim]
        pcdim = pcdimunflat[0] * pcdimunflat[1]
        
        inp_dim = env_enc_dim + output_dim * 2

        if args.planner_type in list_of_cont_planners: 
            dataset = S2DDataset("", "")
        else: #TODO: Make a stepwise dataloader. I need to finish this...
            dataset = S2DStepwiseDataset("", "")

        world_size = 20

        if args.encoder_type == "cnn": 
            env_enc = CNN().to(device)
        elif args.encoder_type == "cnnhch": 
            env_enc = CNNHCH(height=33).to(device)
        elif args.encoder_type == "pnet": 
            pnhiddens = [512, 256, env_enc_dim]
            env_enc = PointNet(output_dim, pnhiddens, output_dim, output_dim, space_dim).to(device) #is correct? Might need to be dim
        elif args.encoder_type == "pnet++": 
            env_enc = PointNet2().to(device)
        elif args.encoder_type == "ndf": 
            env_enc = NDF().to(device)
        elif args.encoder_type == "ff": 
            env_enc = FFEnc(fl_dim=pcdim).to(device)




    elif args.env_type == "ur5cubby": 
        output_dim = 6 
        space_dim = 3
        pcdimunflat = [1024, space_dim] #6? 
        pcdim = pcdimunflat[0] * pcdimunflat[1]

        inp_dim = env_enc_dim + output_dim * 2

        if args.planner_type in list_of_cont_planners: 
            dataset = UR5CubbyDataset("", "")
            val_dataset = UR5CubbyValDataset("", "")
        else: 
            dataset = UR5CubbyStepwiseDataset("", "")
            val_dataset = UR5CubbyStepwiseValDataset("", "")


        world_size_ws = 1 #approximate but very close to accurate, workspace normalization for pc
        world_size_cs = math.pi #exact, model will never have arm ranges outsize +/- math.pi for config space

        if args.encoder_type == "cnn":  #make the 3dCNN model, this will be trickier
            env_enc = CNN3d().to(device)
        elif args.encoder_type == "cnnhch": 
            env_enc = CNNHCH(height=33).to(device)
        elif args.encoder_type == "pnet": 
            pnhiddens = [512, 256, env_enc_dim]
            env_enc = PointNet(output_dim, pnhiddens, output_dim, output_dim, space_dim).to(device) #conf_dim, goal_dim, obstacle_point_dim
        elif args.encoder_type == "pnet++": 
            env_enc = PointNet2().to(device)
        elif args.encoder_type == "ndf": 
            env_enc = NDF3d().to(device)
        elif args.encoder_type == "ff": 
            env_enc = FFEnc(fl_dim=pcdim).to(device)



        #env_enc_dim = 8192 #this is wrong... but idk what this should be?


    lr = args.learning_rate
    num_epochs = args.num_epochs
    tol = args.solver_tolerance


    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=5)
    val_data_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True, num_workers=5)

    #ODEFunc(device, self.hidden_dim, self.obs_dim, self.space_dim, augment_dim, time_dependent, non_linearity)


    if args.planner_type == "rf": 
        net = ResNetFlow(dim, n_layers=n_layers, hidden_dims=[hidden_dim] * n_layers, time_net='TimeLinear')
    elif args.planner_type == "cf": 
        net = CouplingFlow(dim, n_layers=n_layers, hidden_dims=[hidden_dim] * n_layers, time_net='TimeTanh')
    elif args.planner_type == "ff": 
        net = FF(hidden_dim, output_dim).to(device)

    st_enc = nn.Sequential(
            nn.Linear(inp_dim, hidden_dim), 
            nn.ReLU()
            ).to(device)
    st_dec = nn.Sequential(
            nn.ReLU(), 
            nn.Linear(hidden_dim, inp_dim)
            ).to(device)
    pos_dec = nn.Sequential(
            nn.ReLU(), 
            nn.Linear(inp_dim, output_dim)
            ).to(device)

    optimizer = optim.Adam((list(env_enc.parameters()) + list(net.parameters()) + list(st_enc.parameters()) + list(st_dec.parameters()) + list(pos_dec.parameters())), lr=lr)
    opt = optimizer
    mseloss = torch.nn.MSELoss()

    nw = NetWrapper(env_enc, net, st_enc, st_dec, pos_dec, opt)
    nw.cuda()
    #lossfunc = AutonomousMSELoss(device, nw, mseloss, args.time_samples, gamma=0.2)
    lossfunc = mseloss

    if args.start_epoch: 
        fname = os.path.join(args.model_path, 'neuralflow_epoch_%d.pkl' % (args.start_epoch))
        torch_seed, np_seed, py_seed = load_seed(fname)
        torch.manual_seed(torch_seed)
        np.random.seed(np_seed)
        random.seed(py_seed)
        load_net_state(nw, fname)
        load_opt_state(nw, fname)
        optimizer = nw.opt
        nw.to(device)
        env_enc.to(device)
        net.to(device)
        st_enc.to(device)
        st_dec.to(device)
        pos_dec.to(device)



    #world_size = args.world_size
    #init3d(-1, 1, world_res=256) #lets try this... I think I should lower resolution, we only have one point cloud, and this is taking wayy too long
    if args.env_type == "simple2d": 
        init(-1, 1, world_res=256) #TODO: I wanna try some tests with a lower resolution
    elif args.env_type == "ur5cubby": 
        init3d(-1, 1, world_res=33) #lets try this... I think I should lower resolution, we only have one point cloud, and this is taking wayy too long. 64 is still... okay voxelization? 64 per axis? 


    #load saved model
    if args.start_epoch: 
        fname = os.path.join(args.model_path, 'neuralflow_epoch_%d.pkl' % (args.start_epoch))
        torch_seed, np_seed, py_seed = load_seed(fname)
        net.cuda()
        net.to(device)
        '''net.set_opt(optim.Adam, lr=lr)
        load_net_state(func, fname)
        load_opt_state(net, fname)
        load_opt_state(func, fname)'''
    else: #create new model
        net.cuda()
        net.to(device)
        #net.set_opt(optim.Adam, lr=lr)

        torch_seed = np.random.randint(low=0, high=1000)
        np_seed = np.random.randint(low=0, high=1000)
        py_seed= np.random.randint(low=0, high=1000)

    torch.manual_seed(torch_seed)
    np.random.seed(np_seed)
    random.seed(py_seed)

    #start training
    e = 0
    last_e = 0
    while e < num_epochs: 
        batch_avg_loss = 0
        val_mse_batch_avg_loss = 0
        num_batch = 0
        base_dist = dist.uniform.Uniform(0, 1)
        for i, (x_batch, y_batch, path_lens, env_rep) in enumerate(data_loader):
            x_batch = x_batch.type(torch.float32)
            y_batch = y_batch.type(torch.float32)
            optimizer.zero_grad()

            #sample "time_samples" number of eval_times in range [0,1]
            #eval_times, indices = torch.sort(base_dist.sample([batch_size, args.time_samples, 1]), dim=1)
            time_samples = args.time_samples
            #not sure we need this loop anymore, keeping it just in case...
            eval_times = torch.tensor([1, 1, 1])
            while len(eval_times) != len(torch.unique(eval_times)):
                #print("UH OH, DUP EVAL TIME, RESAMPLE")
                #print("eval_times.shape: " + str(torch.unique(eval_times).shape))
                eval_times, indices = torch.sort(base_dist.sample([args.time_samples]))
                eval_range = torch.tensor([0, 1])
                #print("new eval_times.shape: " + str(eval_times.shape))

            eval_times = torch.tensor([eval_range[0], *eval_times, eval_range[1]])
            time_samples += 2



            x_batch = x_batch.to(device)
            if args.planner_type in list_of_cont_planners: 
                fp_x_batch = x_batch[:, 0, :]
            else: 
                fp_x_batch = x_batch
            s_x_batch = fp_x_batch[:, pcdim:(pcdim + output_dim)] #start pt
            g_x_batch = fp_x_batch[:, (pcdim + output_dim):(pcdim + 2 * output_dim)] #goal pt
            if args.planner_type in list_of_cont_planners: 
                s_x_batch = s_x_batch[:, None, :]
                g_x_batch = g_x_batch[:, None, :]

            #random inversion
            '''rng = base_dist.sample([1])
            if  rng[0] < 0.5: 
                print("FLIP!")
                temp = s_x_batch
                s_x_batch = g_x_batch
                g_x_batch = temp'''
            s_g_dists = torch.cdist(s_x_batch, g_x_batch)
            #n_x_batch = b_normalize(x_batch, s_g_dists).to(device)

            if args.env_type == "simple2d": 
                n_x_batch = normalize(x_batch, world_size).to(device) #this works because for simple2d ws and cs are same, so normalize the same
            elif args.env_type == "ur5cubby": 
                n_x_batch_pc = normalize(x_batch[:, :pcdim], world_size_ws).to(device) #normalize pc by workspace dimensions (approximate)
                n_x_batch_sg = normalize(x_batch[:, pcdim:], world_size_cs).to(device) #normalize start/goal by config space dims (exact!)
                n_x_batch = torch.cat((n_x_batch_pc, n_x_batch_sg), dim=1) #TODO probs wrong dimension


            if args.planner_type in list_of_cont_planners: 
                fp_n_x_batch = n_x_batch[:, 0, :]#get first point in traj (start point)
            else: 
                fp_n_x_batch = n_x_batch#get first point in traj (start point)

            env_n_x_batch = fp_n_x_batch[:, :pcdim] #stop hardcoding 2800...
            if args.pc_samps: 
                env_n_x_batch_inds = np.random.choice(env_n_x_batch.shape[1], size=(args.batch_size, args.pc_samps))
                env_n_x_batch = env_n_x_batch[:, env_n_x_batch_inds]
            if args.env_type == "simple2d": 
                if args.encoder_type in list_of_rec_enc: 
                    d_env_n_x_batch = reconstruct_2d_pc(env_n_x_batch) #works!
                elif args.encoder_type in list_of_vox_enc: 
                    r_env = reconstruct_2d_pc(env_n_x_batch) #works!
                    d_env_n_x_batch  = []
                    for env in r_env:
                        d_env_n_x_batch.append(voxelize_pointcloud(env)) #should be batch_size by 2d... occupancy grid? 
                else: 
                    d_env_n_x_batch = env_n_x_batch

            elif args.env_type == "ur5cubby": 
                if args.encoder_type in list_of_rec_enc: 
                    d_env_n_x_batch = reconstruct_3d_pc(env_n_x_batch) #works!
                elif args.encoder_type in list_of_vox_enc: 
                    r_env = reconstruct_3d_pc(env_n_x_batch) #works!
                    d_env_n_x_batch  = []
                    for env in r_env:
                        d_env_n_x_batch.append(voxelize_pointcloud3d(env, res=33)) #should be batch_size by 2d... occupancy grid? 
                else: 
                    d_env_n_x_batch = env_n_x_batch


            d_env_n_x_batch = torch.tensor(d_env_n_x_batch).type(torch.float32).to(device)
            #func.store_envs(d_env_n_x_batch)

            s_n_x_batch = fp_n_x_batch[:, pcdim:(pcdim + output_dim)] #start pt
            g_n_x_batch = fp_n_x_batch[:, (pcdim + output_dim):(pcdim + 2 * output_dim)] #goal pt
            #func.store_goals(g_n_x_batch)

            #linear interpolate y's between target trajectory points
            #this section is obfuscated, but it works, I checked it...
            #sometimes this section bugs out becauseit outputs a 49,2 vector instead of 50,2. Why??????

            '''print("y_batch: " + str(y_batch))
            print("y_batch.shape: " + str(y_batch.shape))'''

            y_batch = torch.tensor(y_batch).to(device)
            if args.planner_type in list_of_cont_planners: 
                interp_ys = np.zeros((y_batch.shape[0], len(eval_times), y_batch.shape[2]))
                for qnd in range(len(y_batch)):  #for each traj in batch
                    ndfroms, nmags = get_norm_mags(y_batch[qnd].cpu(), path_lens[qnd].cpu() + 1) #for each trajectory, get list of: magnitude distances from start point, magnitude distances between points
                    ndfroms = torch.tensor([*ndfroms, 1.0])
                    #targets = torch.tensor(get_targets(y_batch[qnd].cpu(), eval_times.cpu(), ndfroms.cpu(), nmags.cpu(), path_lens[qnd].cpu())) #determine where on trajectory each evaluation time SHOULD land, and return those as targets
                    targets = torch.tensor(get_targets(y_batch[qnd].cpu(), eval_times.cpu(), ndfroms.cpu(), nmags.cpu(), len(y_batch[qnd]) + 1)) #apparently interpolator was always one short? Lets see if this works better, not sure why the increment was required for the interpolator
                    if targets.shape[0] != time_samples: 
                        print("qnd: " + str(qnd))
                        print("y_batch: " + str(y_batch))
                        print("y_batch.shape: " + str(y_batch.shape))
                        print("eval_times: " + str(eval_times))
                        print("eval_times.shape: " + str(eval_times.shape))
                        print("ndfroms: " + str(ndfroms))
                        print("nmags: " + str(nmags))
                        print("targets: " + str(targets))
                        print("targets.shape: " + str(targets.shape))
                    interp_ys[qnd] = targets
                interp_ys = torch.tensor(interp_ys)
                targ_y = torch.tensor(interp_ys).to(device)
                #targ_y = torch.tensor(b_normalize(interp_ys, s_g_dists.cpu())).to(device)
                targ_y = torch.tensor(normalize(interp_ys, world_size_cs)).to(device)

            else: 
                targ_y = torch.tensor(normalize(y_batch, world_size_cs)).to(device)
            
            un_targ_y = unnormalize(targ_y.cpu().detach().numpy(), world_size_cs)


            #might need more data prep?
            #targ_y = normalize(y_batch, world_size)

            #step model (includes loss collection and bprop
            #loss = net.step(torch.tensor(d_env_n_x_batch, dtype=torch.float32).to(device), torch.tensor(s_n_x_batch).to(device), torch.tensor(g_n_x_batch).to(device), targ_y, eval_times=torch.tensor(eval_times).to(device), last_only=False)
            if args.encoder_type in list_of_sg_enc: 
                env_emb = env_enc.forward(d_env_n_x_batch.to(device), s_n_x_batch, g_n_x_batch)
            elif args.encoder_type == "ff": 
                env_emb = env_enc.forward(d_env_n_x_batch.to(device))
            elif args.encoder_type == "cnn" or args.encoder_type == "cnnhch" : 
                '''if args.env_type == "ur5cubby": 
                    d_env_n_x_batch = d_env_n_x_batch[:, 0, :, :]'''
                d_env_n_x_batch.permute(0, 3, 1, 2)
                env_emb = env_enc.forward(d_env_n_x_batch.to(device))
                #env_emb = env_emb[:, 0, :, :]
                env_emb = torch.flatten(env_emb, 1)
            else: 
                env_emb = env_enc.forward(s_n_x_batch.to(device), d_env_n_x_batch.to(device))
            #env_emb = env_enc.abs_encoder_out(d_env_n_x_batch.to(device))
            #env_emb = torch.flatten(env_emb, start_dim=1)
            feats = torch.concat((env_emb, s_n_x_batch, g_n_x_batch), dim=1)
            emb_feats = st_enc(feats)
            emb_feats = torch.reshape(emb_feats, (emb_feats.shape[0], 1, emb_feats.shape[1]))
            c_eval_times = eval_times.repeat(emb_feats.shape[0], 1)  #dup eval_times into shape (batch_size, eval_times)
            c_eval_times = c_eval_times.reshape(emb_feats.shape[0], time_samples, 1)

            if args.planner_type in list_of_cont_planners: 
                n_pred_y = net(emb_feats.to(device), c_eval_times.to(device))
            else: 
                n_pred_y = net(emb_feats.to(device))
            n_pred_feats = st_dec(n_pred_y)
            n_pred_coords = pos_dec(n_pred_feats)

            #evaluate loss
            #loss = mseloss(n_pred_y, targ_y)
            #mse = mseloss(n_pred_coords.float(), targ_y.float()).detach()
            #loss = lossfunc(eval_times, s_n_x_batch, d_env_n_x_batch, g_n_x_batch, n_pred_coords, targ_y)

            np.save("results/sample_pred_path.npy", unnormalize(n_pred_coords.cpu().detach().numpy(), world_size_cs))
                #n_x_batch_sg = normalize(x_batch[:, pcdim:(pcdim + 2 * output_dim)], world_size_cs).to(device) #normalize start/goal by config space dims (exact!)
            np.save("results/sample_gt_path.npy", unnormalize(targ_y.cpu().detach().numpy(), world_size_cs))

            if args.planner_type in list_of_cont_planners: 
                loss = lossfunc(n_pred_coords.float(), targ_y.float())
            else: 
                loss = lossfunc(n_pred_coords[:, 0, :].float(), targ_y[:, :6].float())
            loss.backward()
            optimizer.step()


        for i, (x_val_batch, y_val_batch, path_lens_val, env_rep_val) in enumerate(val_data_loader):
            x_val_batch = x_val_batch.type(torch.float32)
            y_val_batch = y_val_batch.type(torch.float32)
            #optimizer.zero_grad()

            #sample "time_samples" number of eval_times in range [0,1]
            #eval_times, indices = torch.sort(base_dist.sample([batch_size, args.time_samples, 1]), dim=1)
            '''time_samples = args.time_samples
            #not sure we need this loop anymore, keeping it just in case...
            eval_times = torch.tensor([1, 1, 1])
            while len(eval_times) != len(torch.unique(eval_times)):
                #print("UH OH, DUP EVAL TIME, RESAMPLE")
                #print("eval_times.shape: " + str(torch.unique(eval_times).shape))
                eval_times, indices = torch.sort(base_dist.sample([args.time_samples]))
                eval_range = torch.tensor([0, 1])
                #print("new eval_times.shape: " + str(eval_times.shape))

            eval_times = torch.tensor([eval_range[0], *eval_times, eval_range[1]])
            time_samples += 2'''



            x_val_batch = x_val_batch.to(device)
            if args.planner_type in list_of_cont_planners: 
                fp_x_val_batch = x_val_batch[:, 0, :]
            else: 
                fp_x_val_batch = x_val_batch
            s_x_val_batch = fp_x_val_batch[:, pcdim:(pcdim + output_dim)] #start pt
            g_x_val_batch = fp_x_val_batch[:, (pcdim + output_dim):(pcdim + 2 * output_dim)] #goal pt
            if args.planner_type in list_of_cont_planners: 
                s_x_val_batch = s_x_val_batch[:, None, :]
                g_x_val_batch = g_x_val_batch[:, None, :]

            #random inversion
            '''rng = base_dist.sample([1])
            if  rng[0] < 0.5: 
                print("FLIP!")
                temp = s_x_batch
                s_x_batch = g_x_batch
                g_x_batch = temp'''
            #s_g_val_dists = torch.cdist(s_x_val_batch, g_x_val_batch)
            #n_x_batch = b_normalize(x_batch, s_g_dists).to(device)
            if args.env_type == "simple2d": 
                n_x_val_batch = normalize(x_val_batch, world_size).to(device) #this works because for simple2d ws and cs are same, so normalize the same
            elif args.env_type == "ur5cubby": 
                n_x_val_batch_pc = normalize(x_val_batch[:, :pcdim], world_size_ws).to(device) #normalize pc by workspace dimensions (approximate)
                n_x_val_batch_sg = normalize(x_val_batch[:, pcdim:(pcdim + 2 * output_dim)], world_size_cs).to(device) #normalize start/goal by config space dims (exact!)
                #n_x_val_batch_sg = normalize(x_val_batch[:, pcdim:(pcdim + 2 * output_dim)], world_size_cs).to(device) #normalize start/goal by config space dims (exact!)
                n_x_val_batch = torch.cat((n_x_val_batch_pc, n_x_val_batch_sg), dim=1) #TODO probs wrong dimension

            if args.planner_type in list_of_cont_planners: 
                fp_n_x_val_batch = n_x_val_batch[:, 0, :]#get first point in traj (start point)
            else: 
                fp_n_x_val_batch = n_x_val_batch#get first point in traj (start point)


            env_n_x_val_batch = fp_n_x_val_batch[:, :pcdim] #stop hardcoding 2800...
            if args.env_type == "simple2d": 
                if args.encoder_type in list_of_rec_enc: 
                    d_env_n_x_val_batch = reconstruct_2d_pc(env_n_x_val_batch) #works!
                elif args.encoder_type in list_of_vox_enc: 
                    r_env_val = reconstruct_2d_pc(env_n_x_val_batch) #works!
                    d_env_n_x_val_batch  = []
                    for env in r_env_val:
                        d_env_n_x_val_batch.append(voxelize_pointcloud(env)) #should be batch_size by 2d... occupancy grid? 
                else: 
                    d_env_n_x_val_batch = env_n_x_val_batch

            elif args.env_type == "ur5cubby": 
                if args.encoder_type in list_of_rec_enc: 
                    d_env_n_x_val_batch = reconstruct_3d_pc(env_n_x_val_batch) #works!
                elif args.encoder_type in list_of_vox_enc: 
                    r_env_val = reconstruct_3d_pc(env_n_x_val_batch) #works!
                    d_env_n_x_val_batch  = []
                    for env in r_env_val:
                        d_env_n_x_val_batch.append(voxelize_pointcloud3d(env, res=33)) #should be batch_size by 2d... occupancy grid? 
                else: 
                    d_env_n_x_val_batch = env_n_x_val_batch

            d_env_n_x_val_batch = torch.tensor(d_env_n_x_val_batch).type(torch.float32).to(device)
            #func.store_envs(d_env_n_x_batch)

            s_n_x_val_batch = fp_n_x_val_batch[:, pcdim:(pcdim + output_dim)] #start pt
            g_n_x_val_batch = fp_n_x_val_batch[:, (pcdim + output_dim):(pcdim + 2 * output_dim)] #goal pt
            #func.store_goals(g_n_x_batch)

            #linear interpolate y's between target trajectory points
            #this section is obfuscated, but it works, I checked it...
            #sometimes this section bugs out becauseit outputs a 49,2 vector instead of 50,2. Why??????

            '''print("y_batch: " + str(y_batch))
            print("y_batch.shape: " + str(y_batch.shape))
            print("eval_times: " + str(eval_times))
            print("eval_times.shape: " + str(eval_times.shape))'''
            y_val_batch = torch.tensor(y_val_batch).to(device)
            if args.planner_type in list_of_cont_planners: 
                interp_ys_val = np.zeros((y_val_batch.shape[0], len(eval_times), y_val_batch.shape[2]))
                for qnd in range(len(y_val_batch)):  #for each traj in batch
                    ndfroms, nmags = get_norm_mags(y_val_batch[qnd].cpu(), path_lens_val[qnd].cpu()) #for each trajectory, get list of: magnitude distances from start point, magnitude distances between points
                    ndfroms = torch.tensor([*ndfroms, 1.0])
                    #targets = torch.tensor(get_targets(y_batch[qnd].cpu(), eval_times.cpu(), ndfroms.cpu(), nmags.cpu(), path_lens[qnd].cpu())) #determine where on trajectory each evaluation time SHOULD land, and return those as targets
                    targets = torch.tensor(get_targets(y_val_batch[qnd].cpu(), eval_times.cpu(), ndfroms.cpu(), nmags.cpu(), path_lens_val[qnd].cpu() + 1)) #apparently interpolator was always one short? Lets see if this works better, not sure why the increment was required for the interpolator
                    if targets.shape[0] != time_samples: 
                        print("qnd: " + str(qnd))
                        print("y_batch: " + str(y_batch))
                        print("y_batch.shape: " + str(y_batch.shape))
                        print("eval_times: " + str(eval_times))
                        print("eval_times.shape: " + str(eval_times.shape))
                        print("ndfroms: " + str(ndfroms))
                        print("nmags: " + str(nmags))
                        print("targets: " + str(targets))
                        print("targets.shape: " + str(targets.shape))
                    interp_ys_val[qnd] = targets
                interp_ys_val = torch.tensor(interp_ys_val)
                #targ_y_val = torch.tensor(interp_ys_val).to(device)
                #targ_y = torch.tensor(b_normalize(interp_ys, s_g_dists.cpu())).to(device)
                targ_y_val = torch.tensor(normalize(interp_ys_val, world_size_cs)).to(device)
            else: 
                targ_y_val = torch.tensor(normalize(y_val_batch, world_size_cs)).to(device)

            #might need more data prep?
            #targ_y = normalize(y_batch, world_size)

            #step model (includes loss collection and bprop
            #loss = net.step(torch.tensor(d_env_n_x_batch, dtype=torch.float32).to(device), torch.tensor(s_n_x_batch).to(device), torch.tensor(g_n_x_batch).to(device), targ_y, eval_times=torch.tensor(eval_times).to(device), last_only=False)
            if args.encoder_type in list_of_sg_enc: 
                env_emb = env_enc.forward(d_env_n_x_val_batch.to(device), s_n_x_val_batch, g_n_x_val_batch)
            elif args.encoder_type == "ff": 
                env_emb = env_enc.forward(d_env_n_x_val_batch.to(device))
            elif args.encoder_type == "cnn" or args.encoder_type == "cnnhch" : 
                '''if args.env_type == "ur5cubby": 
                    d_env_n_x_batch = d_env_n_x_batch[:, 0, :, :]'''
                d_env_n_x_val_batch.permute(0, 3, 1, 2)
                env_emb = env_enc.forward(d_env_n_x_val_batch.to(device))
                #env_emb = env_emb[:, 0, :, :]
                env_emb = torch.flatten(env_emb, 1)
            else: 
                env_emb = env_enc.forward(s_n_x_val_batch.to(device), d_env_n_x_val_batch.to(device))
            #env_emb = env_enc.abs_encoder_out(d_env_n_x_batch.to(device))
            #env_emb = torch.flatten(env_emb, start_dim=1)
            feats = torch.concat((env_emb, s_n_x_val_batch, g_n_x_val_batch), dim=1)
            emb_feats = st_enc(feats)
            emb_feats = torch.reshape(emb_feats, (emb_feats.shape[0], 1, emb_feats.shape[1]))
            c_eval_times = eval_times.repeat(emb_feats.shape[0], 1)  #dup eval_times into shape (batch_size, eval_times)
            c_eval_times = c_eval_times.reshape(emb_feats.shape[0], time_samples, 1)

            if args.planner_type in list_of_cont_planners: 
                n_pred_y = net(emb_feats.to(device), c_eval_times.to(device))
            else: 
                n_pred_y = net(emb_feats.to(device))
            n_pred_feats = st_dec(n_pred_y)
            n_pred_coords = pos_dec(n_pred_feats)

            #evaluate loss
            #loss = mseloss(n_pred_y, targ_y)
            if args.planner_type in list_of_cont_planners: 
                val_mse = mseloss(n_pred_coords.float(), targ_y_val.float()).detach()
            else: 
                val_mse = mseloss(n_pred_coords[:, 0, :].float(), targ_y_val[:, :6].float()).detach()
            np.save("results/val/pred_path" + str(i) +  ".npy", unnormalize(n_pred_coords.cpu().detach().numpy(), world_size_cs))
                #n_x_batch_sg = normalize(x_batch[:, pcdim:(pcdim + 2 * output_dim)], world_size_cs).to(device) #normalize start/goal by config space dims (exact!)
            np.save("results/val/gt_path" + str(i) + ".npy", unnormalize(targ_y_val.cpu().detach().numpy(), world_size_cs))




            if i % 100 == 99: 
                print("autoloss at iter " + str(i) + ": Training MSE = " + str(loss) + ", Val MSE: " + str(val_mse))
            batch_avg_loss += loss
            val_mse_batch_avg_loss += val_mse
            num_batch += 1
        batch_avg_loss /= num_batch
        val_mse_batch_avg_loss /= num_batch
        e += 1
        print("Epoch " + str(e) + " Avg Train MSE Loss: " + str(batch_avg_loss) + ", Avg Val MSE Loss: " + str(val_mse_batch_avg_loss))
        if (e % args.save_epoch) == 0: 
            if last_e != 0: 
                os.remove(os.path.join(args.model_path, ("neuralflow_epoch_%d.pkl" %(last_e))))
            save_state(nw, torch_seed, np_seed, py_seed, os.path.join(args.model_path, ("neuralflow_epoch_%d.pkl" %(e))))
            last_e = e

             
