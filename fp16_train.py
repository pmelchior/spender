#!/usr/bin/env python
# coding: utf-8

#!pip install git+https://github.com/aliutkus/torchinterp1d.git
#!pip install accelerate
import io, os, sys, time
import numpy as np
import pickle
import torch
from torch import nn
from torch import optim
from torch.distributions.normal import Normal
from torchinterp1d import Interp1d

from util import load_data,skylines_mask,permute_indices
from emission_lines import *
from batch_wrapper import wrap_batches, save_batch, LOGWAVE_RANGE
from model import SpectrumAutoencoder, Instrument

import accelerate
from accelerate import Accelerator, DeepSpeedPlugin
#from memory_profiler import profile
import humanize,psutil,GPUtil

data_dir = "/scratch/gpfs/yanliang"
dynamic_dir = "/scratch/gpfs/yanliang/dynamic-data"
savemodel = "models"

#data_file = ["%s/sdssrand_N74000_spectra.npz"%(data_dir),
#             "%s/bossrand_N74000_spectra.npz"%(data_dir)]

datatag = "chunk256"#"chunk1024"
data_prefix = ["%s%s"%(i,datatag) for i in ["SDSS","BOSS"]]
NBATCH = 300
n_subsample = 50

encoder_names = ["sdss","boss"]
n_encoder = len(encoder_names)
skip=[False,False]

debug = False
option_normalize = True
similarity = True
align_latents = True
annealing_step = 0.05
ANNEAL_SCHEDULE = np.arange(0.2,1,annealing_step)
#ANNEAL_SCHEDULE = np.hstack((ANNEAL_SCHEDULE,np.ones(20)))
print("similarity_slope:",len(ANNEAL_SCHEDULE),ANNEAL_SCHEDULE)


code = "v2"
n_latent = 2

SED = {"data":[True,False], "decoder":True}
BED = {"data":[False,True], "decoder":True}
SE = {"data":[True,False], "decoder":False}
BE = {"data":[False,True], "decoder":False}
D = {"data":[True,True],"encoder":[False,False],"decoder":True}
SEBE = {"data":[True,True], "decoder":False}
FULL = {"data":[True,True],"decoder":True}


def prepare_train(seq,niter=500):
    for d in seq:
        if not "iteration" in d:d["iteration"]=niter
        if not "encoder" in d:d.update({"encoder":d["data"]})
    return seq

train_sequence=prepare_train([FULL])
if "debug" in sys.argv:debug=True
    
model_k = 1
label = "%s/cross-%s"%(savemodel,code)

# model number
# load from
#label_ = label+".%d"%0
label_ ="./models/dataonly-v2.1"

class LogLinearDistribution():
    def __init__(self, a, bound):
        x0,xf = bound
        self.bound = bound
        self.a = a
        self.norm = -a*np.log(10)/(10**(a*x0)-10**(a*xf))
        
    def pdf(self,x):
        pdf = self.norm*10**(self.a*x)
        pdf[(x<self.bound[0])|(x>self.bound[1])] = 0
        return pdf
    
    def cdf(self,x):
        factor = self.norm/(-self.a*np.log(10))
        cdf = -factor*(10**(self.a*x)-10**(self.a*self.bound[0]))
        cdf[x<self.bound[0]] = 0
        cdf[x>self.bound[1]] = 1
        return cdf
    
    def inv_cdf(self,cdf):
        factor = self.norm/(-self.a*np.log(10))
        return (1/self.a)*torch.log10(10**(self.a*self.bound[0])-cdf/factor)
    
    
def collect_batches(datatag,which=None,NBATCH = 100):
    
    data_prefix = ["%s%s"%(i,datatag) for i in ["SDSS","BOSS"]]
    batch_files = os.listdir(dynamic_dir)
    train_batches = []
    valid_batches = []
    test_batches = []
    all_batches = []
    for j in range(len(data_prefix)):  
        batches = [item for item in batch_files if data_prefix[j] in item\
                   and not "copy" in item]
        print("# batches: %d"%len(batches))
        NBATCH = min(NBATCH,len(batches))
        batches = random.sample(batches,NBATCH)
        all_batches.append(batches)
        train_batches.append(batches[:int(0.7*NBATCH)])
        valid_batches.append(batches[int(0.7*NBATCH):int(0.85*NBATCH)])
        test_batches.append(batches[int(0.85*NBATCH):])
    if which == "test": return test_batches
    elif which == "valid": return valid_batches
    elif which == "train": return train_batches
    else: return all_batches

class CPU_Unpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module == 'torch.storage' and name == '_load_from_bytes':
            return lambda b: torch.load(io.BytesIO(b), map_location='cpu')
        else: return super().find_class(module, name)

def load_batch(batch_name):
    with open("%s/%s"%(dynamic_dir,batch_name), 'rb') as f:
        if torch.cuda.is_available():
            batch_copy = pickle.load(f)
        else:batch_copy = CPU_Unpickler(f).load()    
    
    if type(batch_copy)==list:
        spec,w,z = [item.to(device) for item in batch_copy[:3]]
        w[w<=1e-6] = 0
        return spec,w,z
    else:return batch_copy
    
def mem_report():
    print("CPU RAM Free: " + humanize.naturalsize( psutil.virtual_memory().available ))
  
    if torch.cuda.device_count() ==0: return
    
    GPUs = GPUtil.getGPUs()
    for i, gpu in enumerate(GPUs):
        print('GPU {:d} ... Mem Free: {:.0f}MB / {:.0f}MB | Utilization {:3.0f}%'.format(i, gpu.memoryFree, gpu.memoryTotal, gpu.memoryUtil*100))
    return
    
def load_model(fileroot,n_latent=10):
    if not torch.cuda.is_available():
        device = torch.device('cpu')
    else:
        device = None
    
    path = f'{fileroot}.pt'
    print("path:",path)
    model = torch.load(path, map_location=device)
    if type(model)==list or type(model)==tuple:
        [m.eval() for m in model]
    elif type(model)==dict:
        mdict = model
        print("states:",mdict.keys())

        models = mdict["model"]
        instruments = mdict["instrument"]
        model = []
        if "n_latent" in mdict:n_latent=mdict["n_latent"]
        for m in models:
            loadm = SpectrumAutoencoder(wave_rest,n_latent=n_latent,
                                        normalize=option_normalize)
            loadm.load_state_dict(m)
            loadm.eval()
            model.append(loadm)
            
        for ins in instruments:
            empty=Instrument(wave_obs=None, calibration=None)
            model.append(empty)
        
    else: model.eval()
    path = f'{fileroot}.losses.npy'
    loss = np.load(path)
    print (f"model {fileroot}: iterations {len(loss)}, final loss: {loss[-1]}")
    return model, loss
  
def insert_jitters(spec,number,slope=-1.32,bound=[0.0,2]):
    number = int(number)
    location = torch.randint(len(spec), device=device,
                             size=(1,number)).squeeze(0)
    
    loglinear = LogLinearDistribution(slope,bound)
    var = loglinear.inv_cdf(torch.rand(number,device=device))
    amp = var**0.5
    
    # half negative
    half = torch.rand(number)>0.5
    amp[half] = -amp[half]
    
    # avoid inserting jitter to padded regions
    mask = spec[location]>0
    return location[mask],amp[mask]

def jitter_redshift(batch, params, inst):
    # original batch
    spec, w, true_z = batch
    wave_obs = inst.wave_obs
    
    wave_mat = wave_obs*torch.ones_like(spec)
    ncopy = len(params)

    batch_out  = {}
    
    data = []
    
    begin = 0
    # number of copys
    for copy,param in enumerate(params):
        batch_size = len(true_z)
        
        #z_offset,n_lim = param
        z_lim, n_lim = param
        
        # uniform distribution
        z_offset = z_lim*(2*torch.rand(batch_size,device=device)-1)
        
        n_jit = np.random.randint(n_lim[0],n_lim[1], 
                                  size=batch_size)
        
        z_new = true_z+z_offset
        
        z_new[z_new<0] = 0 # avoid negative redshift
        z_new[z_new>0.5] = 0.5 # avoid very high redshift
        
        zfactor = ((1+z_new)/(1+true_z)).unsqueeze(1)*torch.ones_like(spec)

        # redshift linear interpolation
        spec_new = Interp1d()(wave_mat*zfactor,spec,wave_mat)
        w_new = Interp1d()(wave_mat*zfactor,w,wave_mat)
        w_new[w_new<=1e-6] = 0 
        
        record = []
        for i in range(len(spec_new)):
            loc,amp = insert_jitters(spec_new[i],n_jit[i])
            spec_new[i][loc] += amp
            w_new[i][loc] = 1/(amp**2+1/w_new[i][loc])
            record.append([z_offset[i].item(),n_jit[i]])
        
        med = spec_new.median(1,False).values[:,None]
        med[med<1e-1] = 1e-1
        
        spec_new /= med
        #print("median:",spec_new.median(1,False).min())
        
        if torch.isnan(spec_new).any():
            nan_ind = torch.isnan(spec_new)
            print("spec nan! z:", true_z[nan_ind], 
                  "offset:",z_offset[nan_ind])
            
        end = begin+batch_size
        data.append([spec_new,w_new,z_new])
        batch_out[copy]={"param":record,"range":[begin,end]}
        begin=end
    
    new_batch = [torch.cat([d[i] for d in data]) for i in range(3)]
    batch_out['batch'] = new_batch
    return batch_out

def boss_sdss_id():
    save_targets = "/scratch/gpfs/yanliang/headers/joint_headers.pkl"
    f = open(save_targets,"rb")
    targets = pickle.load(f)
    f.close()

    id_names = ['PLATE','MJD','FIBERID']
    sdss_names = ["%s_%s"%(k,"SDSS") for k in id_names]
    boss_names = ["%s_%s"%(k,"BOSS") for k in id_names]

    sdss_id = []
    boss_id = []

    for j in range(len(targets)):
        plate, mjd, fiber = [targets[k][j] for k in boss_names]
        boss_id.append("%d-%d-%d"%(plate, mjd, fiber))

        plate, mjd, fiber = [targets[k][j] for k in sdss_names]
        sdss_id.append("%d-%d-%d"%(plate, mjd, fiber))
        
    return np.array(sdss_id),np.array(boss_id)

def boss2sdss(boss_list):
    sdss_list = []
    for boss in boss_list:
        index = BOSS_id.index(boss)
        sdss_list.append(SDSS_id[index])
    return np.array(sdss_list)

def sdss2boss(sdss_list):
    boss_list = []
    for name in sdss_list:
        index = SDSS_id.index(name)
        boss_list.append(BOSS_id[index])
    return np.array(boss_list)

def build_ladder(train_sequence):
    n_iter = sum([item['iteration'] for item in train_sequence])

    ladder = np.zeros(n_iter,dtype='int')
    n_start = 0
    for i,mode in enumerate(train_sequence):
        n_end = n_start+mode['iteration']
        ladder[n_start:n_end]= i
        n_start = n_end
    return ladder

def get_all_parameters(models,instruments):
    model_params = []
    # multiple encoders
    for model in models:
        model_params += model.encoder.parameters()
    # 1 decoder
    model_params += model.decoder.parameters()
    dicts = [{'params':model_params}]
    
    n_parameters = sum([p.numel() for p in model_params if p.requires_grad])
    
    instr_params = []
    # instruments
    for inst in instruments:
        if inst==None:continue
        instr_params += inst.parameters()
        s = [p.numel() for p in inst.parameters()]
        #print("Adding %d parameters..."%sum(s))
    if instr_params != []:
        dicts.append({'params':instr_params,'lr': 1e-4})
        n_parameters += sum([p.numel() for p in instr_params if p.requires_grad])
        print("parameter dict:",dicts[1])
    return dicts,n_parameters

def augment_loss(model,spec,w,instrument,z,s_true,copy_info,sim_info,mask):
    
    copyname = copy_info.split(".")[0] + "_copy.pkl"
    
    if os.path.isfile("%s/%s"%(dynamic_dir,copyname)):
        print("loading from", copyname)
        batch_copy = load_batch(copyname)
        
    else: 
        print("saving to", copyname)
        batch_copy = jitter_redshift([spec,w,z],mock_params,instrument)
        save_batch(batch_copy,copyname)
        
    
    
    batch_copy = accelerator.prepare(batch_copy)
    spec_copy,w_copy,z_copy = batch_copy["batch"]
    
    if mask != {}:
        # "zero" weight for masked region
        maskmat = mask["mask"].repeat(w_copy.shape[0],1)
        w_copy[maskmat] = 0
    
    # encode the truncated mock data, super-sampled 
    s = model.encode(spec_copy, w=w_copy, z=z_copy)
    if similarity:
        ta = time.time()
        slope = sim_info
        loss = resample_sim(instrument,model,spec_copy,w_copy,
                            z_copy,s,slope=slope)
    else:loss = 0
    tb = time.time()
    print("[copy]similarity_loss: %.3f"%loss.item(),
          "Time: %.2f s"%(tb-ta))
    
    if align_latents:
        # encode original spectra
        loss += augment_similarity(s_true,s)
        tb = time.time()
        print("[copy]similarity+latent loss: %.3f"%loss.item(),
              "Time: %.2f s"%(tb-ta))
        #return loss
    
    
    if torch.isnan(loss).sum()>0:
        print("NAN similarity!!")
        print("s_copy:",torch.isnan(s).sum(),s.min(),s.max())
        print("spec_copy:",torch.isnan(spec_copy).sum(),
              spec_copy.min(),spec_copy.max())
        print("w_copy:",torch.isnan(w_copy).sum(),
              w_copy.min(),w_copy.max())
        print("z_copy:",torch.isnan(z_copy).sum(),
              z_copy.min(),z_copy.max())
        res = resample_sim(instrument,model,
                           spec_copy,w_copy,z_copy,
                           s,verbose=True)
        s_sim,spec_sim,sim_loss = res
        print("s_sim:",s_sim,"spec_sim:",spec_sim,
              "sim_loss:",sim_loss)
        exit()
    
    
    for key in batch_copy:        
        if type(key)==str: continue
        begin,end = batch_copy[key]["range"]
        # redshift to observed frame
        spec_new = model.decoder.transform(\
                   model.decoder.decode(s)[begin:end],instrument,z=z)
        loss += model._loss(spec, w, spec_new)
    
    tb = time.time()
    print("[copy]similarity+latent+spec loss: %.3f"%loss.item(),
          "Time: %.2f s"%(tb-ta))
    
    if debug:
        print("loss_copy:",loss.item())
        if torch.isnan(loss):
            loss_ind = model._loss(spec, w, spec_new, individual=True)
            
            nan_ind = torch.isnan(loss_ind)
            spec_nan = spec_new[nan_ind]
            
            print("loss is nan!", copy_info)
            print("\n nan:",z_copy[nan_ind],
                  spec_nan.max(),spec_nan.min())
            exit() 

    return loss

def augment_similarity(s,s_copy,verbose=False):
    batch_size, s_size = s.shape
    x = torch.sum((s_copy-s)**2/1e-2,dim=1)/s_size
    sim_loss = torch.sigmoid(x)-0.5 # zero = perfect alignment
    if verbose:return x,sim_loss
    else: return sim_loss.sum()

def resample_to_restframe(wave_obs,wave_rest,y,w,z):
    wave_z = (wave_rest.unsqueeze(1)*(1 + z)).T
    wave_obs = wave_obs.repeat(y.shape[0],1)
    # resample restframe spectra on observed spectra
    yrest = Interp1d()(wave_obs,y,wave_z)
    wrest =  Interp1d()(wave_obs,w,wave_z)
    msk = (wave_z<=wave_obs.min())|(wave_z>=wave_obs.max())|(wrest<0)
    yrest[msk]=0
    wrest[msk]=0
    return yrest,wrest

def similarity_loss(spec,w,s,verbose=False,rand=[],wid=5,slope=0.5):
    batch_size, s_size = s.shape
    if rand==[]:rand = permute_indices(batch_size)
    new_w = 1.0/(w**(-1)+w[rand]**(-1))
    D = (new_w > 0).sum(dim=1)
    D[D<1]=1 # avoids NAN in loss function
    spec_sim = torch.sum(new_w*(spec[rand]-spec)**2,dim=1)/D
    s_sim = torch.sum((s[rand]-s)**2,dim=1)/s_size
    x = s_sim-spec_sim
    sim_loss = torch.sigmoid(x)+torch.sigmoid(-slope*x-wid)
    if verbose:return s_sim,spec_sim,sim_loss
    else: return sim_loss.sum()
    
def cross_similarity(spec,w,s,verbose=False,wid=5,slope=0.5):
    batch_size, s_size = s[0].shape
    new_w = torch.zeros_like(w[0])
    msk = (w[0]>0) * (w[1]>0)
    
    new_w[msk] = 1.0/(w[0][msk]**(-1)+w[1][msk]**(-1))
    D = (new_w > 0).sum(dim=1)
    D[D<1]=1 # avoids NAN in loss function
    spec_sim = torch.sum(new_w*(spec[0]-spec[1])**2,dim=1)/D
    s_sim = torch.sum((s[0]-s[1])**2,dim=1)/s_size
    x = s_sim-spec_sim
    sim_loss = torch.sigmoid(x)+torch.sigmoid(-slope*x-wid)
    if verbose:return s_sim,spec_sim,sim_loss
    else: return sim_loss.sum()

def resample_sim(instrument,model,spec,w,z,s,slope=0.5,verbose=False):
    spec,w = resample_to_restframe(instrument.wave_obs,
                                   model.decoder.wave_rest,
                                   spec,w,z)
    sim_loss = similarity_loss(spec,w,s,slope=slope,verbose=verbose)
    return sim_loss
    
def checkpoint(args,optimizer,scheduler,n_encoder,label):
    unwrapped = [accelerator.unwrap_model(args_i).state_dict()\
                 for args_i in args]

    model_unwrapped = unwrapped[:n_encoder]
    instruments_unwrapped = unwrapped[n_encoder:2*n_encoder]

    # checkpoints
    accelerator.save({
        "model": model_unwrapped,
        "instrument":instruments_unwrapped,
        "optimizer": optimizer.optimizer.state_dict(),
        # optimizer is an AcceleratedOptimizer object
        #"scaler": accelerator.scaler.state_dict(),
        "scheduler": scheduler.state_dict(),
        "n_latent":n_latent
    },f'{label}.pt')
    return
    
#@profile
def train(models, accelerator, instruments, train_batches, 
          valid_batches, n_epoch=200, label="", losses = [],
          silent=False, lr=1e-4, fp16=True, data_copy=False,
          mask_skyline=True, similarity=similarity,cross_similar=True):
    
    model_parameters,n_parameters = get_all_parameters(models,instruments)
    
    #torch.autograd.set_detect_anomaly(True)
    print("model parameters:", n_parameters)
    mem_report()
    
    ladder = build_ladder(train_sequence)
    optimizer = optim.Adam(model_parameters, lr=lr, eps=1e-4)
    scheduler = optim.lr_scheduler.OneCycleLR(optimizer, lr, 
                                              total_steps=n_epoch)
    # NEW
    #scaler = torch.cuda.amp.GradScaler()
    
    args = models + instruments + [optimizer]
    prepared = accelerator.prepare(*args)
    models = prepared[:n_encoder]
    instruments = prepared[n_encoder:2*n_encoder]
    optimizer = prepared[-1]
    
    detailed_loss = {}

    if not "train" in detailed_loss:
        detailed_loss["train"]=np.zeros((n_encoder,n_subsample,n_epoch))
    
    if mask_skyline:
        mask_dicts = []
        for which in range(n_encoder):
            locmask = skylines_mask(instruments[which].wave_obs)
            mask_dicts.append({"mask":locmask})
    else: mask_dicts=[{}]*n_encoder
                
    optimizer.zero_grad()
    
    for epoch in range(n_epoch):
        
        loss_ep = np.zeros(5)
        
        mode = train_sequence[ladder[epoch]]
        
        # turn on/off model decoder
        for p in models[0].decoder.parameters():
            p.requires_grad = mode['decoder']
        #print("decoder...",p.requires_grad)
        
        sim_info = ANNEAL_SCHEDULE[epoch%len(ANNEAL_SCHEDULE)]
        print("similarity info:",sim_info)
        
        
        for which in range(n_encoder):
            
            # turn on/off encoder
            for p in models[which].encoder.parameters():
                p.requires_grad = mode['encoder'][which]

            if not mode['data'][which]:continue
            
            models[which].train()
            instruments[which].train()
            
            nsamples = 0
            train_loss = 0.
            nbatch = len(train_batches[which])
            
            # randomly select subsamples in each epoch!!
            sub_samples = random.sample(train_batches[which],
                                        n_subsample)
            print("sub_samples:",sub_samples)
            for k,batchname in enumerate(sub_samples):

                #optimizer.zero_grad()
                batch = load_batch(batchname)
                batch = accelerator.prepare(batch)
                spec, w, z = batch
                
                if mask_skyline:
                    # "zero" weight for masked region
                    maskmat = mask_dicts[which]["mask"].repeat(w.shape[0],1)
                    w[maskmat]=0
                
                batch_size = spec.shape[0]
                nsamples += batch_size
                print("[batch %d/%d,batch_size=%d]: begin"%(k,n_subsample,batch_size))
                mem_report()        

                loss = models[which].loss(spec,w,instruments[which],z=z)
                print("loss_spec:",loss.item())

                if similarity:
                    s = models[which].encode(spec, w=w, z=z)
                    slope = sim_info
                    sim_loss = resample_sim(instruments[which],models[which],
                                            spec,w,z,s,slope=slope)
                    print("sim_loss:",sim_loss.item())
                    print("s: %.3f,%.3f"%
                          (s.min().item(),s.max().item()))
                else: sim_loss=0
                    
                if data_copy:
                    accelerator.backward(loss)  
                    
                else: # skip augments
                    mem_report()
                    ta = time.time()
                    accelerator.backward(loss+sim_loss)  
                    
                    tb = time.time()
                    print("Backward Time: %.2f"%(tb-ta))
                    
                    train_loss += (loss+sim_loss).item()
                    detailed_loss["train"][which][k][epoch] = loss.item()/batch_size
                    
                    torch.nn.utils.clip_grad_norm_(\
                    model_parameters[0]['params'],1.0)
                    optimizer.step()
                    optimizer.zero_grad()
                    
                    tb = time.time()
                    print("Optimizer Step Time: %.2f"%(tb-ta))
                    
                    continue
                    
                args = (models[which],spec,w,instruments[which],
                        z,s,(batchname),sim_info,mask_dicts[which])

                copy_loss = augment_loss(*args)
                copy_loss += sim_loss
                print("copy_loss:",copy_loss)
                batch_loss = 0.5*(loss+copy_loss)
                if np.isnan(batch_loss.item()):
                    print("nan!!")
                    print("spec:",spec.min(),spec.max())
                    print("w:",w.min(),w.max())
                    print("z:",z.min(),z.max())
                    print("s:",s.min(),s.max())
                    exit()

                for p in models[which].parameters():
                    if torch.isnan(p.grad).sum()>0:
                        print("NAN!", p.max(),p.min()) 
                        #exit()
                    else:print("norm:",p.grad.min(),p.grad.max())
                        
                train_loss += batch_loss.item()
                
                detailed_loss["train"][which][k][epoch] = batch_loss.item()
                
                mem_report()
                accelerator.backward(copy_loss)  
                torch.nn.utils.clip_grad_norm_(\
                model_parameters[0]['params'],1.0)
                # once per batch
                optimizer.step()
                optimizer.zero_grad()
            
            train_loss /= nsamples
            loss_ep[n_encoder*which] = train_loss
            
        # SDSS/BOSS cross similarity term ??
        cross_loss = 0
        if cross_similar:
            sub_samples = [random.sample(train_batches[which],n_subsample)\
                           for which in range(n_encoder)]
                
            for k in range(n_subsample):
                batchnames = [sub_samples[which][k]\
                              for which in range(n_encoder)]
                
                s = []
                specs = []
                ws = []
                ta = time.time()
                for which,batchname in enumerate(batchnames):
                    batch = load_batch(batchname)
                    spec, w, z = accelerator.prepare(batch)
                    #batch_size = spec.shape[0]
                    
                    spec,w = resample_to_restframe(\
                             instruments[which].wave_obs,
                             models[which].decoder.wave_rest,
                             spec,w,z)
                    
                    s.append(models[which].encode(spec, w=w, z=z))
                    specs.append(spec)
                    ws.append(w)
                    
                sim_loss = cross_similarity(specs,ws,s,verbose=False)
                
                if np.isnan(sim_loss.item()):
                    print("nan!!")
                    print("spec:",spec.min(),spec.max())
                    print("w:",w.min(),w.max())
                    print("z:",z.min(),z.max())
                    print("s[0]",s[0].min(),s[0].max(),
                          s[1].min(),s[1].max())
                    exit()
                
                tb = time.time()
                print("cross similarity Time: %.2f"%(tb-ta))
                print("sim_loss:",sim_loss.item())       
                accelerator.backward(sim_loss)  
                cross_loss += sim_loss.item()
                
                tb = time.time()
                print("Backward Call Time: %.2f"%(tb-ta))
                torch.nn.utils.clip_grad_norm_(\
                model_parameters[0]['params'],1.0)
                optimizer.step()
                optimizer.zero_grad()
            
            # cross loss
            cross_loss /= nsamples
            loss_ep[-1] = cross_loss
                
        scheduler.step()
        with torch.no_grad():
            for which in range(n_encoder):
                if skip[which]:continue
                models[which].eval()
                instruments[which].eval()
            
                valid_loss = 0.
                nsamples = 0
                nbatch = len(valid_batches[which])
                for k,batchname in enumerate(valid_batches[which]):
                    batch = load_batch(batchname)
                    batch = accelerator.prepare(batch)
                    spec, w, z = batch
                    batch_size = spec.shape[0]
                    nsamples += batch_size
                    if mask_skyline:
                        # "zero" weight for masked region
                        maskmat = mask_dicts[which]["mask"].repeat(w.shape[0],1)
                        w[maskmat]=0
                    #with torch.cuda.amp.autocast(enabled=fp16):
                    loss = models[which].loss(spec,w, instruments[which],z=z)
                    if similarity:
                        s = models[which].encode(spec, w=w, z=z)
                        
                        slope = sim_info
                        sim_loss = resample_sim(instruments[which],models[which],
                                                spec,w,z,s,slope=slope)
                    else: sim_loss=0
                    
                    if data_copy:
                        args = (models[which],spec,w,instruments[which],
                                z,s,(batchname),sim_info,mask_dicts[which])
                        copy_loss = augment_loss(*args)
                        valid_loss += 0.5*(loss.item()+copy_loss.item()+sim_loss.item())
                    else:valid_loss += (loss+sim_loss).item()
                valid_loss /= nsamples
                
                loss_ep[n_encoder*which+1] = valid_loss
                if not mode['data'][which]:
                    loss_ep[n_encoder*which] = valid_loss
        losses.append(loss_ep)
        
        if not silent:
            print('====> Epoch: %i TRAINING Loss: %.2e,%.2e VALIDATION Loss: %.2e,%.2e' % (epoch, loss_ep[0], loss_ep[2],  loss_ep[1], loss_ep[3]))
        #exit()
        
        if epoch % 5 == 0 or epoch == n_epoch - 1:           
            
            save_loss = np.array(losses)   
            np.save(f'{label}.losses.npy', save_loss)
            np.save(f'{label}.detail.npy', detailed_loss["train"])

            args = models + instruments
            checkpoint(args,optimizer,scheduler,n_encoder,label)
    return

# load data, specify GPU to prevent copying later
# TODO: use torch.cuda.amp.autocast() for FP16/BF16 typcast
# TODO: need dynamic data loader if we use larger data sets

# restframe wavelength for reconstructed spectra
# Note: represents joint dataset wavelength range
lmbda_min = 2359#data['wave'].min()/(1+data['z'].max())
lmbda_max = 10402#data['wave'].max()
bins = 7000#int(data['wave'].shape[0] * (1 + data['z'].max()))
wave_rest = torch.linspace(lmbda_min, lmbda_max, bins, dtype=torch.float32)

#print ("Observed frame:\t{:.0f} .. {:.0f} A ({} bins)".format(wave_obs.min(), wave_obs.max(), len(wave_obs)))
print ("Restframe:\t{:.0f} .. {:.0f} A ({} bins)".format(lmbda_min, lmbda_max, bins))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("torch.cuda.device_count():",torch.cuda.device_count())

if "new" in sys.argv: 
    models = []
    loss = []
    lr = 1e-3
    n_hidden = (64, 256, 1024)
    for i in range(n_encoder):
        if skip[i]:continue
        model_i = SpectrumAutoencoder(wave_rest,
                                      n_latent=n_latent,
                                      n_hidden=n_hidden,
                                      normalize=option_normalize)
        models.append(model_i)
        print("model_i.normalize:",model_i.normalize)
    # reuse decoder
    if len(models)==2:
        print("Using the same decoder!!")
        models[1].decoder = models[0].decoder
    instruments = [None]*n_encoder
    #exit()
else: 
    
    models, loss = load_model(label_,n_latent=n_latent)
    instruments = list(models[n_encoder:])
    models = list(models[:n_encoder])
    
    if len(models)==2:
        print("Using the same decoder!!")
        models[1].decoder = models[0].decoder
        
    loss  = [list(ll) for ll in loss]
    lr = 1e-4

SDSS_id, BOSS_id =  boss_sdss_id()
data_ids =  [SDSS_id, BOSS_id]
n_ids, n_dim = len(data_ids[0]),2

#accelerator = Accelerator()
accelerator = Accelerator(mixed_precision='fp16')

#deepspeed_plugin = DeepSpeedPlugin(zero_stage=2, gradient_accumulation_steps=2)
#accelerator = Accelerator(fp16=True, deepspeed_plugin=deepspeed_plugin)

uniform_njit = [100,300]
mock_params = [[0.4,uniform_njit]]#,[0.1,uniform_njit]]
ncopy = len(mock_params)


batch_files = os.listdir(dynamic_dir)
train_batches = []
valid_batches = []
wave_obs = []

import random
random_seed = 42
random.seed(random_seed)

alltrain = collect_batches(datatag,which="train",NBATCH=NBATCH)
jointtrain = collect_batches("joint256",which="train",NBATCH=NBATCH)
allvalid = collect_batches(datatag,which="valid",NBATCH=NBATCH)
jointvalid = collect_batches("joint256",which="valid",NBATCH=NBATCH)

for j in range(n_encoder):  
    lower,upper = LOGWAVE_RANGE[j]
    wave_obs.append(10**torch.arange(lower, upper, 0.0001))
    #wrap_batches(j,datatag,k_range=[0,30])

    train_sample = alltrain[j]+jointtrain[j]
    train_sample = random.sample(train_sample,len(train_sample))
    
    valid_sample = allvalid[j]+jointvalid[j]
    valid_sample = random.sample(valid_sample,len(valid_sample))
    
    train_batches.append(train_sample)
    valid_batches.append(valid_sample)
    
    if instruments[j]==None:
        print("Initializing instrument...")
        print("wave_obs:",len(wave_obs[j]))
        instruments[j]=Instrument(wave_obs[j], calibration=None)
    else:# for compatibility
        instruments[j].wave_obs = wave_obs[j]
        
print("train:",train_batches)#,"valid:",valid_batches)
#exit()
n_epoch = sum([item['iteration'] for item in train_sequence])
print ("--- Model %d"%model_k)

init_t = time.time()
train(models, accelerator, instruments, train_batches, 
      valid_batches, n_epoch=n_epoch, losses = loss,
      label=label+f".{model_k}", lr=lr)

print("--- %s seconds ---" % (time.time()-init_t))
