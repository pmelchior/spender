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

from utils import *
from model import *
from accelerate import Accelerator
#from memory_profiler import profile
import humanize,psutil,GPUtil


data_dir = "/scratch/gpfs/yanliang"
dynamic_dir = "/scratch/gpfs/yanliang/dynamic-data"
savemodel = "models"

#data_file = ["%s/cutted-sdss_spectra.npz"%(data_dir),
#             "%s/boss-20k_spectra.npz"%(data_dir)]

data_file = ["%s/joint-sdss_spectra.npz"%(data_dir),
             "%s/joint-boss_spectra.npz"%(data_dir)]

encoder_names = ["sdss"]#,"boss"]
n_encoder = len(encoder_names)
skip=[False,True]

debug = False
bat_prefix = "normalized"#"randz"
copy_exists = True
#calib_active = [False,True]

SED = {"data":[True,False], "decoder":True}
BE = {"data":[False,True], "decoder":False}
D = {"data":[True,True],"encoder":[False,False],"decoder":True}
SEBE = {"data":[True,True], "decoder":False}
#FULL = {"data":[True,True],"decoder":True}
model_K = 32
n_redundant = 3

def prepare_train(seq,niter=800):
    for d in seq:
        if not "iteration" in d:d["iteration"]=niter
        if not "encoder" in d:d.update({"encoder":d["data"]})
    return seq

train_sequence=prepare_train([SED])
if "debug" in sys.argv:debug=True

model_k = 0
#code = "sdss100k"
code = "redundancy"

label = "%s/robust-%s"%(savemodel,code)
#save_copy = "%s/robust-copy.%d.pt"%(savemodel,model_k)

# model number
# load from
label_ = label+".%d"%model_k
#label_ = "%s/robust-sdss100k.1"%savemodel
#label_ = "%s/robust-%s.%d"%(savemodel,"test",5)

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

class CPU_Unpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module == 'torch.storage' and name == '_load_from_bytes':
            return lambda b: torch.load(io.BytesIO(b), map_location='cpu')
        else: return super().find_class(module, name)

def load_batch(k,data_file,mode="train"):
    dataname = os.path.basename(data_file).split(".")[0]
    batch_name = "[%s]%s-%s-%d.pkl"%(bat_prefix,dataname,mode,k)
    
    with open("%s/%s"%(dynamic_dir,batch_name), 'rb') as f:
        if torch.cuda.is_available():batch_copy = pickle.load(f)
        else:batch_copy = CPU_Unpickler(f).load()
    return batch_copy

def save_batch(batch_copy,k,data_file,mode="train"):
    dataname = os.path.basename(data_file).split(".")[0]
    batch_name = "[%s]%s-%s-%d.pkl"%(bat_prefix,dataname,mode,k)
    with open("%s/%s"%(dynamic_dir,batch_name), 'wb') as f:
        pickle.dump(batch_copy,f)
    print("Saving to %s/%s.."%(dynamic_dir,batch_name))
    return

def mem_report():
    print("CPU RAM Free: " + humanize.naturalsize( psutil.virtual_memory().available ))
  
    if torch.cuda.device_count() ==0: return
    
    GPUs = GPUtil.getGPUs()
    for i, gpu in enumerate(GPUs):
        print('GPU {:d} ... Mem Free: {:.0f}MB / {:.0f}MB | Utilization {:3.0f}%'.format(i, gpu.memoryFree, gpu.memoryTotal, gpu.memoryUtil*100))
    return
    
def load_model(fileroot):
    if not torch.cuda.is_available():
        device = torch.device('cpu')
    else:
        device = None

    path = f'{fileroot}.pt'
    model = torch.load(path, map_location=device)
    if type(model)==list or type(model)==tuple:
        [m.eval() for m in model]
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
        w_new[w_new<1e-6] = 1e-6 # avoid zero uncertainty
        
        record = []
        for i in range(len(spec_new)):
            loc,amp = insert_jitters(spec_new[i],n_jit[i])
            spec_new[i][loc] += amp
            w_new[i][loc] = 1/(amp**2+1/w_new[i][loc])
            record.append([z_offset[i],n_jit[i]])
        
        med = spec_new.median(1,False).values[:,None]
        med[med<1e-1] = 1e-1
        
        spec_new /= med
        #print("median:",spec_new.median(1,False).min())
        
        if torch.isnan(spec_new).any():
            nan_ind = torch.isnan(spec_new)
            print("spec nan! z:", true_z[nan_ind], 
                  "offset:",z_offset[nan_ind])
            #exit()
            
        end = begin+batch_size
        data.append([spec_new,w_new,z_new])
        batch_out[copy]={"param":record,"range":[begin,end]}
        begin=end
    
    new_batch = [torch.cat([d[i] for d in data]) for i in range(3)]
    batch_out['batch'] = new_batch
    
    return batch_out

def boss_sdss_id():
    save_targets = "joint_headers.pkl"
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

#@profile
def latent_loss(model,spec,w,instrument,z,copy_info,nbatch,
                lambda_latent=1):
    
    #with torch.cuda.amp.autocast():
    s,_,spectrum_observed = model._forward(spec, w, instrument, z)
    loss = model._loss(spec, w, spectrum_observed)
    # load partial copies!!
    ratio = 1
    lambda_latent /= ratio
    if copy_info[0]/nbatch>ratio:return loss

    batch_copy = load_batch(*copy_info)
    #batch_copy = accelerator.prepare(batch_copy)
    spec_copy,w_copy,z_copy = batch_copy["batch"]
    s_copy = model.encode(spec_copy,w_copy)
    
    if debug:
        print("spec:",spec[0].min(),spec[0].max())
        print("z copy:",z_copy[:5])
        print("spec_copy:",spec_copy[0].min(),spec_copy[0].max())
        print("w_copy:",w_copy[0].min())
        print("s:",s.shape,s)
        print("s_copy:",s_copy.shape,s_copy)
        print("\n\nspec loss:",loss.item())
        print("lambda_latent",lambda_latent)
    loss_lat = 0
    for key in batch_copy:
        if type(key)==str: continue
        begin,end = batch_copy[key]["range"]
        loss_lat += torch.sum((s_copy[begin:end]-s).pow(2))
    
    print("loss_lat: ",loss_lat.dtype)   
    
    if debug:print("latent loss:",lambda_latent*loss_lat.item())
    #exit()
    return loss+lambda_latent*loss_lat

def robust_loss(model,spec,w,instrument,z,copy_info,nbatch):
    
    batch_copy = load_batch(*copy_info)
    batch_copy = accelerator.prepare(batch_copy)
    
    spec_copy,w_copy,z_copy = batch_copy["batch"]
    
    # encode the truncated mock data, super-sampled 
    #w_copy = None
    print("spec_copy:",spec_copy.shape)
    s = model.encode(spec_copy, w=w_copy)
    
    spec_rest = model.decoder.decode(s)
    
     # variations of data copies
    loss = 0
    for key in batch_copy:        
        if type(key)==str: continue

        begin,end = batch_copy[key]["range"]
        
        #data window redshifted to z=0
        wave_z0 = (instrument.wave_obs.unsqueeze(1)/(1 + z)).T
        wave_rest = model.decoder.wave_rest.repeat(end-begin,1)
        
        # resample model on restframe data
        spec_new = Interp1d()(wave_rest,spec_rest[begin:end],wave_z0)
        
        loss += model._loss(spec, w, spec_new)
    
    if debug:
        print("loss_copy:",loss.item())
        if torch.isnan(loss):
            loss_ind = model._loss(spec, w, spec_new, individual=True)
            
            nan_ind = torch.isnan(loss_ind)
            spec_nan = spec_new[nan_ind]
            
            print("loss is nan!", copy_info)
            print("\n nan:",z_copy[nan_ind],spec_nan.max(),spec_nan.min())
            exit() 

    return loss

#@profile
def train(models, accelerator, instruments, trainloaders, 
          validloaders, n_epoch=200, label="", losses = [],
          silent=False, lr=1e-4, fp16=True):
    
    model_parameters,n_parameters = get_all_parameters(models,instruments)
    
    print("model parameters:", n_parameters)
    mem_report()
    
    ladder = build_ladder(train_sequence)
    
    optimizer = optim.Adam(model_parameters, lr=lr)
    scheduler = optim.lr_scheduler.OneCycleLR(optimizer, lr, 
                                              total_steps=n_epoch)
    # NEW
    scaler = torch.cuda.amp.GradScaler()

    
    args = models + instruments + [optimizer]
    prepared = accelerator.prepare(*args)
    models = prepared[:n_encoder]
    instruments = prepared[n_encoder:2*n_encoder]
    optimizer = prepared[-1]
    
    
    for epoch in range(n_epoch):
        
        loss_ep = np.zeros(4)
        
        mode = train_sequence[ladder[epoch]]
        
        # turn on/off model decoder
        for p in models[0].decoder.parameters():
            p.requires_grad = mode['decoder']
        #print("decoder...",p.requires_grad)
        
        for which in range(n_encoder):
            
            # turn on/off encoder
            for p in models[which].encoder.parameters():
                p.requires_grad = mode['encoder'][which]

            if not mode['data'][which]:continue
            
            models[which].train()
            instruments[which].train()
            
            train_loss = 0.
            nbatch = len(trainloaders[which].dataset)//batch_size
            for k,batch in enumerate(trainloaders[which]):
                optimizer.zero_grad()
                spec, w, z = batch

                print("[batch %d/%d]: begin"%(k,nbatch))
                mem_report()
                
                # ignore weight?
                loss = models[which].loss(spec,w,instruments[which],z=z)
                
                #spectrum_observed = models[which].forward(spec, w=None, instrument=instruments[which], z=z)
                #loss = models[which]._loss(spec, w, spectrum_observed)
                print("loss_spec:",loss.item())
                scaler.scale(loss).backward()
                
                
                # back-propagation!!
                with torch.cuda.amp.autocast(enabled=fp16):
                    args = (models[which],spec,w,instruments[which],
                            z,(k,data_file[which]),nbatch)
                    copy_loss = robust_loss(*args)
                    #copy_loss = loss
                # NEW
                scaler.scale(copy_loss).backward()
                
                mem_report()
                #accelerator.backward(loss)
                train_loss += 0.5*(loss.item()+copy_loss.item())

            train_loss /= (len(trainloaders[which].dataset))
            loss_ep[n_encoder*which] = train_loss
        
        # NEW
        scaler.step(optimizer)
        scaler.update()
        
        #optimizer.step()
        scheduler.step()

        with torch.no_grad():
            for which in range(n_encoder):
                if skip[which]:continue
                models[which].eval()
                instruments[which].eval()
            
                valid_loss = 0.
                nbatch = (len(validloaders[which].dataset)//batch_size+1)
                for k,batch in enumerate(validloaders[which]):
                    spec, w, z = batch
                    
                    loss = models[which].loss(spec,w, instruments[which],z=z)
                    
                    
                    with torch.cuda.amp.autocast(enabled=fp16):
                        args = (models[which],spec,w,instruments[which],
                                z,(k,data_file[which],"valid"),nbatch)
                        copy_loss = robust_loss(*args)
                    
                    valid_loss += 0.5*(loss.item()+copy_loss.item())
                    
                    
                valid_loss /= (len(validloaders[which].dataset))
                
                loss_ep[n_encoder*which+1] = valid_loss
                if not mode['data'][which]:
                    loss_ep[n_encoder*which] = valid_loss
            
        losses.append(loss_ep)
        
        if epoch % 10 == 0 or epoch == n_epoch - 1:           
            if not silent:
                print('====> Epoch: %i TRAINING Loss: %.2e,%.2e VALIDATION Loss: %.2e,%.2e' % (epoch, loss_ep[0], loss_ep[2],  loss_ep[1], loss_ep[3]))

            save_loss = np.array(losses)
            # checkpoints
            torch.save(models+instruments, f'{label}.pt')
            
            np.save(f'{label}.losses.npy', save_loss)
            
            #return

            
# load data, specify GPU to prevent copying later
# TODO: use torch.cuda.amp.autocast() for FP16/BF16 typcast
# TODO: need dynamic data loader if we use larger data sets

# restframe wavelength for reconstructed spectra
# Note: represents joint dataset wavelength range
lmbda_min = 2371#data['wave'].min()/(1+data['z'].max())
lmbda_max = 10402#data['wave'].max()
bins = 7000#int(data['wave'].shape[0] * (1 + data['z'].max()))
wave_rest = torch.linspace(lmbda_min, lmbda_max, bins, dtype=torch.float32)

#print ("Observed frame:\t{:.0f} .. {:.0f} A ({} bins)".format(wave_obs.min(), wave_obs.max(), len(wave_obs)))
print ("Restframe:\t{:.0f} .. {:.0f} A ({} bins)".format(lmbda_min, lmbda_max, bins))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("torch.cuda.device_count():",torch.cuda.device_count())

if "new" in sys.argv: 
    model = []
    loss = []
    lr = 1e-3
    n_latent = 10
    n_hidden = (1024, 256, 64)

    for i in range(n_encoder):
        if skip[i]:continue
        model_i = SpectrumAutoencoder(wave_rest,
                                      n_latent=n_latent,
                                      n_hidden=n_hidden,
                                      K=model_K,
                                      n_redundant=n_redundant)
        model.append(model_i)
    # reuse decoder
    if len(model)==2:model[1].decoder = model[0].decoder
    instruments = [None]*n_encoder
    
else: 
    
    models, loss = load_model(label_)
    model = list(models[:n_encoder])
    instruments = list(models[n_encoder:])
    
    loss  = [list(ll) for ll in loss]
    lr = 1e-5

SDSS_id, BOSS_id =  boss_sdss_id()
data_ids =  [SDSS_id, BOSS_id]
n_ids, n_dim = len(data_ids[0]),10

batch_size=512

trainloaders=[]
validloaders=[]

accelerator = Accelerator()


for j in range(len(data_file)):
    if skip[j]:continue
    
    print("Loading dataset %s"%data_file[j])
    
    data=load_data(data_file[j], which="train", device=device)
    valid_data=load_data(data_file[j], which="valid", device=device)
    wave_obs = torch.tensor(data['wave'], dtype=torch.float32)
    wave_obs = wave_obs.to(accelerator.device)
    
    
    trainloader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(data['y'], data['w'], 
        data['z']), batch_size=batch_size, 
        shuffle=False)

    validloader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(valid_data['y'], 
        valid_data['w'], valid_data['z']),
        batch_size=batch_size)
    
    trainloader, validloader = accelerator.prepare(trainloader, 
                                                   validloader)
    trainloaders.append(trainloader)
    validloaders.append(validloader)
    
    if instruments[j]==None:
        print("Initializing instrument...")
        print("wave_obs:",len(wave_obs))
        
        # polynomial calibration
        instruments[j]=Instrument(wave_obs, calibration=None)
    else:# for compatibility
        instruments[j].wave_obs = wave_obs
    
    # only renew copies at new run
    #if not "new" in sys.argv:continue
    if copy_exists:continue
    
    # prepare copies!!!
    uniform_njit = [100,300]
    params = [[0.4,uniform_njit]]#,[0.1,uniform_njit]]
    ncopy = len(params)

    for k, batch in enumerate(trainloader):
        spec, w, z = batch  
        
        print("\nInitializing batch %d copies"%k)
        ta = time.time()
        
        batch_copy = jitter_redshift(batch,params,instruments[j])
        save_batch(batch_copy,k,data_file[j],mode="train")
        tb = time.time()
        
    for k, batch in enumerate(validloader):
        spec, w, z = batch  
        
        print("\nInitializing batch %d copies"%k)
        ta = time.time()
        batch_copy = jitter_redshift(batch,params,instruments[j])
        save_batch(batch_copy,k,data_file[j],mode="valid")
        tb = time.time()
                    


n_epoch = sum([item['iteration'] for item in train_sequence])

print ("--- Model %d"%model_k)

init_t = time.time()

train(model, accelerator, instruments, trainloaders, 
      validloaders, n_epoch=n_epoch, losses = loss,
      label=label+f".{model_k}", lr=lr)

print("--- %s seconds ---" % (time.time()-init_t))
