#!/usr/bin/env python
# coding: utf-8
import io, os, sys,time,random
import numpy as np
import pickle
import torch
from torchinterp1d import Interp1d
from util import load_data,skylines_mask,permute_indices
from emission_lines import *
from batch_wrapper import wrap_batches, save_batch, LOGWAVE_RANGE
from model import SpectrumAutoencoder, Instrument
#from memory_profiler import profile
import humanize,psutil,GPUtil
import matplotlib.pyplot as plt

data_dir = "/scratch/gpfs/yanliang"
dynamic_dir = "/scratch/gpfs/yanliang/dynamic-data"
savemodel = "models"
#device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device =  torch.device("cpu")

# restframe wavelength for reconstructed spectra
# Note: represents joint dataset wavelength range
lmbda_min = 2359#data['wave'].min()/(1+data['z'].max())
lmbda_max = 10402#data['wave'].max()
bins = 7000#int(data['wave'].shape[0] * (1 + data['z'].max()))
wave_rest = torch.linspace(lmbda_min, lmbda_max, bins, dtype=torch.float32)
print ("Restframe:\t{:.0f} .. {:.0f} A ({} bins)".format(lmbda_min, lmbda_max, bins))

def merge(merge_dir,name=''):
    
    imfiles = os.listdir(merge_dir)
    
    imfiles = [f for f in imfiles if ".png" in f]
    if name != '':
        imfiles = [f for f in imfiles if name in f]
    imfiles = sorted(imfiles)
    
    im_list = []
    for filename in imfiles:
        im = Image.open("%s/%s"%(merge_dir,filename))
        im.load() # required for png.split()

        copy = Image.new("RGB", im.size, (255, 255, 255))
        copy.paste(im, mask=im.split()[3]) # 3 is the alpha channel
        
        im_list.append(copy)
    
    n = len(imfiles)
    pdf_name = "[%s]%s-%dpage.pdf"%(name,merge_dir,n)

    im1=im_list[0]
    print("Saving %d pages to %s"%(len(im_list),pdf_name))
    im1.save(pdf_name, "PDF" ,resolution=100.0, 
             save_all=True, append_images=im_list[1:])
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

def sdss_name(plate, mjd, fiberid):
    flocal = os.path.join(dat_dir, 'sdss-spectra/spec-%s-%i-%s.fits' % (str(plate).zfill(4), mjd, str(fiberid).zfill(4)))
    return flocal

def add_emission(ax,z=0, ymax=0, xlim=(3000,7000),alpha=0.3,
                 color="grey",ytext=0.5,zorder=5):
    
    lines = [i for i in emissionlines]
    
    if not ymax:ymax = ax.get_ylim()[1]
    
    ymin = -10
    shifted = np.array(lines)*(1+z)
    ax.vlines(x=shifted,ymin=ymin,ymax=1.1*ymax,
              color=color,lw=1,alpha=alpha,zorder=zorder)
    
    for line in emissionlines:
        
        name = emissionlines[line]
        shifted = np.array(line)*(1+z)
        
        if shifted<xlim[0] or shifted>xlim[1]:continue
            
        ax.text(shifted,ytext,name,fontsize=20,
                color=color,rotation=90,zorder=zorder)
                
    return

def boss_sdss_id():
    save_targets = "joint_headers.pkl"
    f = open("%s/headers/%s"%(data_dir,save_targets),"rb")
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
        
    return sdss_id,boss_id

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
        
def plot_loss(loss, ax=None,xlim=None,ylim=None,fs=15):
    latest = loss[:2,-1]
    ep = range(len(loss[0]))
    
    if not ax:fig,ax=plt.subplots(dpi=200)
    labels = ["Train SDSS","Valid SDSS","Train BOSS","Valid BOSS"]
    colors = ['k','r','b','orange']
    
    minimum = np.min(loss[:2])
    
    final_ep = loss[:,-1]
    for i in range(len(loss)):
        if sum(loss[i])==0:continue
        ax.semilogy(ep,loss[i],label="%s(loss=%.2f)"%(labels[i],final_ep[i]),
                                                      color=colors[i])
        
    #ax.axhline(0.5,ls="--",color="b",label="loss = 0.5")
    ax.set_xlabel("epoch")
    ax.set_ylabel("loss")
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.legend(loc="best",fontsize=fs)
    
    if min(latest)==minimum:
        print("%.2f is the minimum!"%minimum)
    return

def find_intersection(ids_sdss,ids_boss):
    joint_ids = []
    for ii in ids_boss:
        if not ii in BOSS_id: continue   
        sdss_key = boss2sdss([ii])[0]
        if not sdss_key in ids_sdss:continue
        joint_ids.append(sdss_key)
    print("intersection:",len(joint_ids))
    return joint_ids

def concatenate(samples,dim=10):
    ls = [len(s) for s in samples]
    print(ls)
    keys = []
    
    s_matrix = torch.zeros((sum(ls),dim))
    begin = 0
    for i,sample in enumerate(samples):
        end = begin+ls[i]
        key = np.arange(begin,end)
        s_matrix[key]=sample
        keys.append(key)
        begin=end
    return s_matrix,keys

def draw_line(marks, ax, lw=1):
    mark_sdss,mark_boss = marks
    for i in range(mark_sdss.shape[1]):
        #if not i in max_ind:continue
        locx = np.vstack((mark_sdss[:,i],mark_boss[:,i])).T
        ax.plot(locx[0],locx[1],'k-',lw=lw,alpha=1)
    return


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
        return (1/self.a)*np.log10(10**(self.a*self.bound[0])-cdf/factor)
    
def insert_jitters(spec,number,slope=-1.32,bound=[0.0,2]):
    number = int(number)
    location = np.random.randint(len(spec), size=number)
    
    loglinear = LogLinearDistribution(slope,bound)
    var = loglinear.inv_cdf(torch.rand(number))
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
    
    wave_mat = wave_obs*torch.ones_like(spec)#wave_obs.unsqueeze(1).T
    print("wave_mat:",wave_mat.shape)#,wave_mat[:5][:5])

    batch_out  = {}
    
    # number of copys
    for copy,param in enumerate(params):
        
        z_offset,n_lim = param
        
        n_jit = np.random.randint(n_lim[0],n_lim[1], 
                                  size=len(true_z))
        
        z_new = true_z+z_offset
        z_new[z_new<0] = 0 # avoid negative redshift
        
        zfactor = ((1+z_new)/(1+true_z)).unsqueeze(1)*torch.ones_like(spec)

        # redshift linear interpolation
        spec_new = Interp1d()(wave_mat*zfactor,spec,wave_mat)
        w_new = Interp1d()(wave_mat*zfactor,w,wave_mat)
        
        record = []
        for i in range(len(spec_new)):
            loc,amp = insert_jitters(spec_new[i],n_jit[i])
            spec_new[i][loc] += amp
            w_new[i][loc] = 1/(amp**2+1/w_new[i][loc])
            record.append([z_offset,n_jit[i]])
        batch_out[copy]={"param":record,"batch":[spec_new,w_new,z_new] }
    return batch_out

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
        w[w<1e-6] = 1e-6
        return spec,w,z,batch_copy[3],batch_copy[4]
    else:return batch_copy
    
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

def merge_batch(file_batches):
    spectra = [];weights = [];z = [];specid = [];norm = []
    
    for batchname in file_batches:
        batch = load_batch(batchname)
        spectra.append(batch[0])
        weights.append(batch[1])
        z.append(batch[2])
        specid.extend(batch[3])
        norm.extend(batch[4])
        
    spectra = torch.cat(spectra,axis=0)
    weights = torch.cat(weights,axis=0)
    z = torch.cat(z,axis=0)
    print("spectra:",spectra.shape,"w:",weights.shape,
          "z:",z.shape)
    return spectra, weights, z, specid, norm

def _normalize(x, m, w=None):
    # apply constant factor c that minimizes (c*m - x)^2
    if w is None:
        w = 1
    mw = m*w
    c = (mw * x).sum(dim=-1) / (mw * m).sum(dim=-1)
    return m * c.unsqueeze(-1), c
    
def model_forward(model,x, w=None, instrument=None, z=None, s=None):
    # normalized!
    if s is None:
        s = model.encode(x, w=w, z=z)
    spectrum_restframe = model.decode(s)
    spectrum_observed = model.decoder.transform(spectrum_restframe, instrument=instrument, z=z)
    if model.normalize:
        spectrum_observed,coeff = _normalize(x, spectrum_observed, w=w)
        print("coeff: %.2f"%coeff.item())
        spectrum_restframe*=coeff.unsqueeze(-1)
    return s, spectrum_restframe, spectrum_observed

def fake_jitter(params, instruments, which, offset=3,
                w_factor = [], flat_weight=[], locid=[],
                fix_z=False, prim_xlim=[],
                model_free=[],plot=True,axs_list=None):
    colors = ['mediumseagreen',"skyblue",'salmon','orange','gold']
    code = ["SDSS","BOSS"]
    
    which_sdss,which_boss = which
    
    if which_boss==[]:
        ydata=[spec_sdss[which_sdss]]
        waves = [ins_sdss.wave_obs]
        weights = [w_sdss[which_sdss]]
        true_z = [z_sdss[which_sdss]]
        titles = ["SDSS:%s"%ids_sdss[which_sdss[0]]]
    else:
        ydata=[spec_sdss[which_sdss],spec_boss[which_boss]]
        waves = [ins_sdss.wave_obs,ins_boss.wave_obs]
        weights = [w_sdss[which_sdss],w_boss[which_boss]]
        true_z = [z_sdss[which_sdss],z_boss[which_boss]]
        titles = ["SDSS:%s"%ids_sdss[which_sdss[0]],
                  "BOSS:%s"%ids_boss[which_boss[0]]]
    
    n_test = len(params)
    print(n_test)
    
    if w_factor ==[]: w_factor = np.ones(n_test)
    if model_free ==[]: model_free = [True]*n_test
    if flat_weight ==[]: flat_weight = [False]*n_test
    
    ndata = len(ydata)
    embed = torch.zeros((ndata,n_test+1,s_size))
    flat_ydata = []
    mean_weights = []
    rest_models = []
    
    # two dataset
    for j in range(ndata):
        loss_total = 0
        #if ndata>1:ax = axs[j]
        axs=axs_list[j]
        ax = axs[0]
        
        recon = torch.zeros((n_test+1,len(instruments[j].wave_obs)))
        recon_w = torch.zeros((n_test+1,len(instruments[j].wave_obs)))
        recon_z = torch.zeros(n_test+1)

        for i,param in enumerate(params):
            z_new,n_jit = param
            wrest = inspect_mix[j].decoder.wave_rest
            wave_data = waves[j]/(1+true_z[j])
            
            zfactor = (1+z_new)/(1+true_z[j])
            locwave = waves[j][None,:]
            
            # define recon_ij
            if model_free[i]:
                # no artifitial redshift
                if fix_z:
                    recon_ij = torch.clone(ydata[j])
                    fake_w = torch.clone(weights[j])
                # apply artifitial redshift
                else:
                    # redshift linear interpolation
                    recon_ij = Interp1d()(locwave*zfactor,ydata[j],locwave)
            # model based spectra
            else:
                clean_model = inspect_mix[j].forward(ydata[j], weights[j], instruments[j],torch.tensor([z_new]).float())
                recon_ij = torch.clone(clean_model)
            
            # define fake_w
            if flat_weight[i]:
                
                fake_w =  weights[j].mean()*torch.ones_like(recon_ij)
                fake_w[weights[j]<2e-6] = 1e-6
            else: fake_w = Interp1d()(locwave*zfactor,weights[j],locwave)
            
            if locid != []:
                if locid[i]=="nothing + true w":
                    recon_ij = torch.ones_like(recon_ij)
                if "permute w" in locid[i]:
                    print("permuting weights! size =", fake_w.shape[1])
                    reorder = permute_indices(fake_w.shape[1])
                    fake_w[0] = fake_w[0][reorder]
                    fake_w[weights[j]<2e-6] = 1e-6

                    
            loc,amp = insert_jitters(recon_ij[0],n_jit)
            recon_ij[0][loc] += amp
            fake_w[0][loc] = 1/(amp**2+1/fake_w[0][loc])
            
            
            fake_w*=w_factor[i]
            print("z=%.2f weight mean:%.2f weight std:%.2f"%(z_new,fake_w.mean(),fake_w.std()))
            print("recon_ij: %.2f +/= %.2f"%(recon_ij.mean(),
                  recon_ij.std()))
            # model arguments
            args = (recon_ij.float(), fake_w.float(), instruments[j],
                    torch.tensor([z_new]).float())
            
            if inspect_mix[j].normalize:
                latent,y_rest,_ = model_forward(inspect_mix[j],*args)
            else:latent,y_rest,_ = inspect_mix[j]._forward(*args)
            
            embed[j][i] = latent[0]
            recon[i] = recon_ij[0]
            recon_w[i] = fake_w[0]
            recon_z[i] = z_new
            
            spec_new = inspect_mix[j].decoder.transform(\
                       y_rest,instruments[j],z=true_z[j])
            wave_new = instruments[j].wave_obs/(1 + true_z[j])
            
            loss = inspect_mix[j]._loss(ydata[j], weights[j], spec_new)
            loss_total += loss
            
            if not plot:continue

            loc_off = offset*(i+1)
            y_rest = y_rest[0].detach()+loc_off
            
            
            disp_wave,disp_ij,disp_w = waves[j]/(1+z_new),recon_ij,fake_w
            
            yloc = disp_ij.detach().numpy()[0] +loc_off
            y_err = (disp_w[0]**(-0.5)).numpy()
            y_err[y_err>0.5*offset]=0.5*offset
            y1 = yloc-y_err;y2 = yloc+y_err

            if locid ==[]: label = "N=%d, z=%.2f (loss=%.2f)"%(n_jit,z_new,loss)
            else: label = "%s (loss=%.2f)"%(locid[i],loss)
            ylabel = "augmented: %.2f +/= %.2f"%(recon_ij.mean(),recon_ij.std())
            
            ax.fill_between(disp_wave,y1,y2,step='mid',
                            color=colors[i],alpha=0.5)
            ax.plot(disp_wave,yloc,c=colors[i],drawstyle='steps-mid',
                    label=ylabel)
            
            ax.plot(wrest,y_rest,drawstyle='steps-mid', c='k',lw=0.5)

            axs[1].plot(wave_new,spec_new.detach()[0],c=colors[i],
                        lw=3, zorder=0, drawstyle='steps-mid',label=label)
            axs[2].plot(wave_new,spec_new.detach()[0],c=colors[i],
                        lw=1, zorder=0, drawstyle='steps-mid',label=label)
            
            
        wj = weights[j]
        y_clone = ydata[j].clone()
        
        args = (y_clone, wj, instruments[j], true_z[j])
        print("forward data... true w mean: %.2f, std: %.2f"%(wj.mean(),wj.std()))
        
        if inspect_mix[j].normalize:
                latent,y_rest,_ = model_forward(inspect_mix[j],*args)
        else:latent,y_rest,_ = inspect_mix[j]._forward(*args)
        loss = inspect_mix[j].loss(*args)
        
        y_rest = y_rest[0].detach()
        
        embed[j][-1] = latent[0]
        recon[-1] = ydata[j][0]
        recon_w[-1] = wj[0]
        recon_z[-1] =  true_z[j]
        
        mean_loss = loss_total/len(params)
        print("%s mean weight: %.2f  mean loss: %.2f"%\
              (code[j],weights[j].mean().item(),mean_loss))
        
        mean_weights.append(weights[j].mean().item())
        if not plot:continue
        
        y_err=wj[0]**(-0.5)
        y_err[y_err>0.5*offset]=0.5*offset
        
        title = titles[j]
        print(title)

        flat_ydata.append(y_clone[0].detach())#ydata[j].detach()[0])
        ylabel = "true: %.2f +/= %.2f"%\
        (flat_ydata[j].mean(),flat_ydata[j].std())
        
        print("flat_ydata[j]: %.2f +/- %.2f"%(flat_ydata[j].mean(),     flat_ydata[j].std()))
        wh_ext = torch.argmax(flat_ydata[j])
        print("extrema:", wh_ext,"wave:",wave_data[wh_ext])
        
        #print("flat_ydata:",flat_ydata[j].min(),flat_ydata[j].max())

        ax.fill_between(wave_data,flat_ydata[j]-y_err,flat_ydata[j]+y_err,color="k",step='mid',alpha=0.2,zorder=-10)
        ax.plot(wave_data,flat_ydata[j],"k-",drawstyle='steps-mid',
                label=ylabel)
        ax.plot(wrest,y_rest,drawstyle='steps-mid', c='r',
                label="true loss: %.2f"%loss,lw=0.5)
        
        
        
        ax.set_xlim(prim_xlim)
        ax.set_ylim(-1.5*offset,(n_test+1)*offset)
        #ax.legend(title=title,loc='best')
        ax.set_xlabel("restframe $\lambda (\AA)$")
        ax.set_title("(z=%.2f) %s"%(true_z[j],title))
        
        
        y_err=wj[0]**(-0.5)
        axs[1].set_title("(z=%.2f) %s"%(true_z[j],title))
        axs[1].fill_between(wave_data,flat_ydata[j]-y_err,flat_ydata[j]+y_err,
                            color="k",step='mid',alpha=0.2,zorder=-10)
        axs[1].plot(wave_data,flat_ydata[j],"k-",lw=2,
                    drawstyle='steps-mid')
        
        axs[1].plot(wrest,y_rest,"r-",lw=2,drawstyle='steps-mid',
                    label="true loss: %.2f"%loss)#,lw=0.5)
        axs[1].set_xlabel("restframe $\lambda (\AA)$")
        axs[2].set_title("(z=%.2f) %s"%(true_z[j],title))
        axs[2].fill_between(wave_data,flat_ydata[j]-y_err,flat_ydata[j]+y_err,
                            color="k",step='mid',alpha=0.2,zorder=-10)
        axs[2].plot(wave_data,flat_ydata[j],"k-",lw=1,zorder=-10,
                    drawstyle='steps-mid')
        
        axs[2].plot(wrest,y_rest,"r-",lw=1,drawstyle='steps-mid',
                    label="true loss: %.2f"%loss)#,lw=0.5)
        
        axs[2].set_xlabel("restframe $\lambda (\AA)$")

        # compare models
        rest_models.append(y_rest)
        
        xlim = (6530,6600)
        msk = (wave_data>xlim[0])*(wave_data<xlim[1])
        axs[1].set_xlim(xlim)
        if len(flat_ydata[j][msk])>0:axs[1].set_ylim(0.3,1.2*flat_ydata[j][msk].max().item())
        #print("y_rest[msk].max():",y_rest[msk].max())
        add_emission(axs[1],z=0, ymax=1.4*y_rest.max(),xlim=xlim)#, ymax=0)
        
        #if prim_xlim == []:
        xlim = (wave_data.min(),wave_data.max())
        #else: xlim = prim_xlim
        msk = (wrest>xlim[0])*(wrest<xlim[1])
        axs[2].set_xlim(xlim[0],xlim[1])
        axs[2].set_ylim(-0.5,min(2.5,1.2*y_rest[msk].max()))
        #axs[2].legend(title=title,loc='best')
        axs[0].legend()
        
        axs[2].legend(loc=4)
        #if plot:plt.tight_layout()
    
        print("recon shape:",recon.shape)
        # evaluate similarity loss
        dim = recon.shape[0]
        rand = [dim-1]*dim
        
        recon,recon_w = resample_to_restframe(waves[j],wrest,
                                              recon,recon_w,recon_z)
        s_sim,spec_sim,sim_loss = similarity_loss(recon,recon_w,embed[j],
                                                  verbose=True,rand=rand)
        print("s_chi:",s_sim)
        print("spec_chi:",spec_sim)
        print("similarity loss:",sim_loss)
    
    
    cross_loss = []
    # evaluate SDSS vs BOSS loss
    # interpolate to SDSS
    wave_z0 = (instruments[0].wave_obs.unsqueeze(1)).T
    boss_model = Interp1d()(waves[1],ydata[1],wave_z0)
    S2B_data_ratio = ydata[0]/boss_model
    
    loss = inspect_mix[0]._loss(ydata[0], weights[0], boss_model)
    cross_loss.append(loss.item())
    
    # interpolate to BOSS
    wave_msk = (waves[1]>waves[0].min())*(waves[1]<waves[0].max()).unsqueeze(0)
    wave_z0 = (waves[1][wave_msk[0]].unsqueeze(1)).T
    sdss_model = Interp1d()(waves[0],ydata[0],wave_z0)
    #print("wave_msk:", wave_msk.shape,"ydata[1]:",ydata[1].shape,"weights[1]:",weights[1].shape)
    loss = inspect_mix[1]._loss(ydata[1][wave_msk].unsqueeze(0), weights[1][wave_msk].unsqueeze(0), sdss_model)
    cross_loss.append(loss.item())
    
    S2B_ratio = rest_models[0]/rest_models[1]
    print("S2B ratio:",S2B_ratio.shape,S2B_ratio)
    other = {"S2B":S2B_ratio,"S2B_data_ratio":S2B_data_ratio}
    return embed,mean_loss,cross_loss,mean_weights,other

def plot_latent(embed, true_z, axs, titles=[], locid=[]):
    if titles==[]:titles=[""]*n_encoder
    if locid==[]:locid = ["z=%.2f"%n for n in newz] + ["true z=%.2f"%true_z[j]]
    # plot latent variables
    for j in range(n_encoder):
        ax = axs[j]
        s = embed[j].detach().numpy()
        # visualize latent space
        ax.matshow(s,vmax=vmax,vmin=vmin)#,cmap='winter')
        
        title="%s"%(titles[j])
        
        for i in range(len(locid)):
            ax.text(1.7,i+0.2,locid[i],c="k", weight='bold',fontsize=25)
        ax.set_xlabel("latent variables");ax.set_ylabel("copies")
        ax.set_title(title,weight='bold',fontsize=30)
        ax.set_xticks([]);ax.set_yticks([])
    return

def plot_embedding(embedded,color_code,name,cmap=None, c_range=[0,0.5], zorder=[0,0],
                   alpha = [1,1], cbar_label="$z$",comment=[],color=None,markers=None,ax=None,mute=False):
    labels = ['SDSS','BOSS']
    if not markers:markers = ["o",'^']
    
    ms = 50;lw = 1
    
    # color-coded backgrounds
    embedding_sdss,embedding_boss = embedded[0]
    
    if color_code==[]:
        color_code = ['salmon','skyblue']
    img = ax.scatter(embedding_sdss[0],embedding_sdss[1], edgecolors='salmon',zorder=zorder[0],
                     s=ms, vmin=c_range[0],vmax=c_range[1],c=color_code[0],label=labels[0],
                     alpha=alpha[0],cmap=cmap)
    
    ax.scatter(embedding_boss[0],embedding_boss[1],s=ms,vmin=c_range[0],vmax=c_range[1], 
               edgecolors='b',c=color_code[1],alpha=alpha[1],cmap=cmap,label=labels[1],zorder=zorder[1])

    cbar = plt.colorbar(img)
    cbar.set_label(cbar_label)
    
    
    nbatch = len(embedded)-1
    if len(comment)==0:comment=[{}]*nbatch
    if not color:color = [['salmon','skyblue']]*nbatch
    
    for k in range(nbatch):
        batch = embedded[k+1]
        if len(batch)==2:draw_line(batch,ax)
        for j in range(len(batch)):
            ax.plot(batch[j][0],batch[j][1],label="%s(%s)"%(labels[j],name[k]),ms=10,lw=0,
                    markeredgewidth=1,markeredgecolor='k',
                    color=color[k][j], marker=markers[k],alpha=1)
            
            args=comment[k]
            if not "ID" in args: continue
            if args["wh"] != j: continue
            
            if not "pos" in args: args["pos"]=(0,0)
            if not "fs" in args: args["fs"]=15  
            
            for ii in range(len(batch[j][0])):
                ax.annotate(args["ID"][ii],(batch[j][0][ii],batch[j][1][ii]), color='k', 
                            fontsize=args["fs"],textcoords="offset points",xytext=args["pos"],
                            ha='right', bbox=dict(boxstyle="round,pad=0.3", lw=0, fc='w',alpha=0.8))
                if mute:continue
                print("sdss vs. boss")
                print(args["ID"][ii],batch[j][0][ii],batch[j][1][ii])
            
    ax.legend(loc=4)
    return


def resample_to_restframe(wave_obs,wave_rest,y,w,z):
    wave_z = (wave_rest.unsqueeze(1)*(1 + z)).T
    wave_obs = wave_obs.repeat(y.shape[0],1)
    # resample restframe spectra on observed spectra
    yrest = Interp1d()(wave_obs,y,wave_z)
    wrest =  Interp1d()(wave_obs,w,wave_z)
    msk = (wave_z<=wave_obs.min())|(wave_z>=wave_obs.max())
    yrest[msk]=0
    wrest[msk]=1e-6
    return yrest,wrest
    i=0
    print("y:",y.shape,"yrest:",yrest.shape,"w:",w.shape,"wrest:",wrest.shape)
    fig,ax = plt.subplots(figsize=(8,5))
    ax.plot(wave_obs[0],y[i],"k-",label="yobs")
    ax.plot(wave_rest*(1+z[i]),yrest[i],"b-",label="yrest")
    #ax.set_xlim(5000,6000)
    ax.legend()
    plt.savefig("resample.png",dpi=120)
    exit()
    return yrest,wrest

# translates spectra similarity to latent space similarity
def similarity_loss(spec,w,s,verbose=False,rand=[],wid=5,slope=0.5):
    batch_size, s_size = s.shape
    if rand==[]:rand = permute_indices(batch_size)
    new_w = 1/(w**(-1)+w[rand]**(-1))
    D = (new_w > 1e-6).sum(dim=1)
    spec_sim = torch.sum(new_w*(spec[rand]-spec)**2,dim=1)/D
    s_sim = torch.sum((s[rand]-s)**2,dim=1)/s_size
    x = s_sim-spec_sim
    #x = torch.sqrt(s_sim)-torch.sqrt(spec_sim)
    sim_loss = torch.sigmoid(x)+torch.sigmoid(-slope*x-wid)
    if verbose:return s_sim,spec_sim,sim_loss
    else: return sim_loss.sum()

def prepare_data(view,wave,mask_skyline = True):
    intensity_limit = 2;radii = 5
    data = merge_batch(view)
    if mask_skyline:
        locmask = skylines_mask(wave)
        maskmat = locmask.repeat(data[1].shape[0],1)
        data[1][maskmat]=1e-6
    return data
#-------------------------------------------------------
from batch_wrapper import LOGWAVE_RANGE

n_encoder = 2

instruments = []
for j in range(n_encoder):
    lower,upper = LOGWAVE_RANGE[j]
    wave_obs = 10**torch.arange(lower, upper, 0.0001)
    inst = Instrument(wave_obs,calibration=None)
    instruments.append(inst)

option_normalize = True
model_file = "models/dataonly-v2.1"
#model_file = "models/anneal-v2.5"
inspect_mix, loss = load_model("%s"%(model_file))
ins_sdss,ins_boss = instruments

random_seed = 42
random.seed(random_seed)

testset = collect_batches("all",which="test",NBATCH=25)
view = collect_batches("joint",which=None)#,which="test")
view[0] += testset[0]
view[1] += testset[1]

wave_sdss = instruments[0].wave_obs
wave_boss = instruments[1].wave_obs

#batch_copy = load_batch("SDSSchunk1024_199680_copy.pkl")
#spec_sdss,w_sdss,z_sdss = [item.to(device) for item in #batch_copy["batch"][:3]]
spec_sdss,w_sdss,z_sdss,ids_sdss,norm_sdss = \
prepare_data(view[0],wave_sdss)
spec_boss,w_boss,z_boss,ids_boss,norm_boss = \
prepare_data(view[1],wave_boss)

    
SDSS_id, BOSS_id =  boss_sdss_id()
joint_ids = find_intersection(ids_sdss,ids_boss)
wh_joint_sdss = [ids_sdss.index(i) for i in joint_ids]
wh_joint_boss = [ids_boss.index(i) for i in sdss2boss(joint_ids)]

names = ["SDSS spectra", "BOSS spectra"]
specs = [spec_sdss,spec_boss]
weights = [w_sdss, w_boss]
redshifts = [z_sdss, z_boss]
lognorms = [np.log10(norm_sdss), np.log10(norm_boss)]
#interesting = [14019,4207,4839,121,6776]


if "model" in sys.argv:
    # visualize 10 latent space
    fig,axs = plt.subplots(1,3,figsize=(8,4),dpi=200,constrained_layout=True,
                           gridspec_kw={'width_ratios': [1.5, 1, 1]})

    loss = loss.T
    epoch = loss.shape[1]
    print("\n\n",model_file)
    plot_loss(loss,ax=axs[0],fs=10)
    
    np.random.seed(23)#42 
    rand = np.random.randint(len(wh_joint_sdss), size=10)
    #rand = np.arange(10)

    for j in range(n_encoder):
        ax=axs[j+1]

        inds = np.array([wh_joint_sdss,wh_joint_boss][j])[rand]
        zlist = redshifts[j][inds].detach().numpy()
        rank = np.argsort(zlist)
        inds = inds[rank]

        s, _, _ = inspect_mix[j]._forward(specs[j][inds], weights[j][inds],
                                          instruments[j],redshifts[j][inds])
        s = s.detach().numpy()

        batch_size,s_size = s.shape
        locid = ["z=%.2f(%d)"%(n,ii) for ii,n in zip(rand[rank],redshifts[j][inds])] 

        vmin,vmax=s.min(),s.max()
        print("smin,smax:",vmin,vmax)
        ax.matshow(s,vmax=vmax,vmin=vmin)#,cmap='winter')
        for i in range(len(locid)):
            ax.text(2,i+0.2,locid[i],c="k", weight='bold', fontsize=12,
                    alpha=1)
        ax.set_xlabel("latent variables");ax.set_ylabel(names[j])
        ax.set_xticks([]);ax.set_yticks([])

    #axs[0].set_ylim(0.5,1)
    axs[0].set_title("($n_{latent}=%d$) best loss = %.2f"%(s_size,min(loss[1])))
    plt.savefig("check_model.png",dpi=200)
    
    if "alone" in sys.argv:exit()
    
# background points
np.random.seed(42)
N = 2000
rand1 = np.random.randint(len(ids_sdss), size=N)
rand2 = np.random.randint(len(ids_boss), size=N)

s_sdss, _, _ = inspect_mix[0]._forward(spec_sdss[rand1], w_sdss[rand1], ins_sdss,z_sdss[rand1])
s_boss, _, _ = inspect_mix[1]._forward(spec_boss[rand2], w_boss[rand2], ins_boss,z_boss[rand2])
print("[Random sample] SDSS:",s_sdss.shape)#,"BOSS:",s_boss.shape)
vmin,vmax=s_sdss.min(),s_sdss.max()
s_size = s_sdss.shape[1]

if "sim" in sys.argv:
    fig,ax = plt.subplots(figsize=(8,6))
    s_sim,spec_sim,sim_loss =similarity_loss(\
    spec_sdss[rand1],w_sdss[rand1],s_sdss,verbose=True)
    img = ax.scatter(spec_sim.detach().cpu(),s_sim.detach().cpu(),
                     c=sim_loss.detach(),cmap="plasma")
    print("sim_loss:",sim_loss.sum())
    s_sim,spec_sim,sim_loss = similarity_loss(\
    spec_boss[rand2],w_boss[rand2],s_boss,verbose=True)
    ax.scatter(spec_sim.detach().cpu(),s_sim.detach().cpu(),
                     c=sim_loss.detach(),cmap="plasma")
    print("sim_loss:",sim_loss.sum())
    cbar = fig.colorbar(img, ax=ax)
    cbar.set_label("loss")
    ax.set_ylabel("latent chi^2")
    ax.set_xlabel("spectral chi^2")
    ax.set_xlim(0,30)
    ax.set_ylim(0,5)
    plt.savefig("similaity.png",dpi=120)
    exit()

# define augmented samples 
nparam = 3
w_factor = [];use_data=[];flat_weight=[];number = np.zeros(nparam);
newz = np.linspace(0,0.5,nparam)

#newz = np.array([-1e-2,1e-2])
#number = np.array([0,0])
locid = ["z=%.1f"%(zz) for zz,n in zip(newz,number)]
locid += ["raw data"]

#print(newz)
#newz+=0.21601364016532898#0.4863503873348236
newz[newz<0]=0;newz[newz>0.5]=0.5
#number = np.random.randint(100,300, size=len(newz))#[0,500,1000]#
params = np.array([newz,number]).T

interesting = [2,3,4,5,474,617,5234]#[2,3,4,5,6]# 185,1,65
for wh_number in interesting:
    wh_sdss = [wh_joint_sdss[wh_number]]
    wh_boss = [wh_joint_boss[wh_number]]

    titles = names
    which = [wh_sdss,wh_boss]
    true_z = [z_sdss[which[0]].item(),z_boss[which[1]].item()]
    print("true z:",true_z)

    #fig,axs = plt.subplots(ncols=ndata,nrows=3)
    fig = plt.figure(constrained_layout=True,figsize=(24,16),dpi=200)
    gs = fig.add_gridspec(4, 4, width_ratios=[1,1,0.5,0.5],hspace=0.01)
    ax1 = fig.add_subplot(gs[:2, 0])
    ax2 = fig.add_subplot(gs[:2, 1])

    ax3 = fig.add_subplot(gs[0, 2])
    ax4 = fig.add_subplot(gs[0, 3])
    ax10 = fig.add_subplot(gs[1, 2:])
    ax5 = fig.add_subplot(gs[2, 0])
    ax6 = fig.add_subplot(gs[3, 0])
    ax7 = fig.add_subplot(gs[2, 1])
    ax8 = fig.add_subplot(gs[3, 1])

    ax9 = fig.add_subplot(gs[2:4,2:4])
    sdss_axs = [ax1,ax5,ax7]
    boss_axs = [ax2,ax6,ax8]
    axs = [sdss_axs,boss_axs]

    embed,mean_loss,cross_loss,mean_weights,other = \
    fake_jitter(params,instruments,which, model_free=use_data,
                flat_weight=flat_weight,locid=locid,
                w_factor=w_factor,fix_z=False,offset=10,
                prim_xlim=None,#[2385,2450],#[6800,7000],#[6500,6600],
                axs_list=[sdss_axs,boss_axs])

    test_embed = [item for item in embed]
    sdss_track,boss_track = embed[0][-1][None],embed[1][-1][None]

    basic = [s_sdss,s_boss,sdss_track,boss_track]
    s_mat, keys = concatenate(basic+test_embed,dim=s_size)
    s_mat = s_mat.detach().numpy()

    # 2D!!
    embedding = s_mat
    embedded = [embedding[key].T for key in keys]
    embedded_batch = [embedded[:2],embedded[2:4],embedded[4:]]

    plot_latent(embed,true_z,[ax3,ax4],titles=titles,locid=locid)
    colors = [['salmon','skyblue'],['salmon','skyblue'],['orange',"navy"],['gold','azure']]
    name = ["target","#jitter+redshift"]

    embed_color=[z_sdss[rand1],z_boss[rand2]]
    c_range = [0,0.5];cmap="gist_rainbow";cbar_label = "$z$"

    mark_id = np.array(ids_sdss)[wh_sdss]
    comment = [{"ID":mark_id,"wh":1},
               {"pos":(-5,0),"ID":locid+[None],"wh":1}]
    plot_embedding(embedded_batch,embed_color,name=name,comment=comment,
                   cbar_label=cbar_label, c_range=c_range,
                   markers=["o","^","s"],color=colors, ax=ax9, mute=True, cmap=cmap)
    
    #xlim = (-0.2,5);ylim = (-0.2,6)
    xlim = (-0.2,7);ylim = (-0.2,7)
    
    ax9.set_xlim(xlim);ax9.set_ylim(ylim)

    text = "Data similarity:\nSDSS/BOSS as model: %.2f vs. %.2f\n"%(cross_loss[1],cross_loss[0])
    text += "SDSS/BOSS weight: %.2f vs. %.2f"%(mean_weights[0],mean_weights[1])
    ax9.text(0.3*xlim[1],0.8*ylim[1],text,fontsize=20)

    ax = ax10
    wrest = wave_rest;msk = (wrest>3000)*(wrest<7500)
    ax.plot(wave_sdss/(1+true_z[0]),other['S2B_data_ratio'][0],"k-",label="data")
    ax.plot(wrest,other['S2B'],"r-",label='model')
    ax.set_xlabel("restframe $\lambda (\AA)$")
    ax.set_ylabel("SDSS/BOSS")
    ax.set_xlim((3000,7000))
    ax.set_ylim(0,1.5)
    ax.legend(loc=4)
    plt.savefig("[%d]z=%.2f.png"%(wh_number,true_z[0]),dpi=200)


if "skip" in sys.argv: exit()

n_joint = 200
rand = np.array(random.sample(range(len(wh_joint_sdss)), n_joint))
s_track = []

mark_ids = rand
for j in range(n_encoder):
    inds = np.array([wh_joint_sdss,wh_joint_boss][j])[rand]
    s, _, _ = inspect_mix[j]._forward(specs[j][inds], weights[j][inds],
                                      instruments[j],redshifts[j][inds])
    s = s.detach().numpy().T
    s_track.append(s)

diff = np.sqrt(np.sum((s_track[0]-s_track[1])**2, axis=0))
med_diff = np.median(diff)

dist_rank = np.argsort(diff)[::-1]
diff = diff[dist_rank]
mask_names = mark_ids[dist_rank]

text = "Rank   No.   Distance\n"
for ii in range(10):
    text += "%d         %d       %.2f\n"%(ii+1,mask_names[ii],diff[ii])
text += "(median distance = %.3f)"%(med_diff)
colors = [['gold','navy']]

embedded_batch = [embedded[:2],s_track]
comment = [{"ID":mark_ids,"wh":0, "pos":(-10,0),"fs":5}]
#comment = []

fig,ax = plt.subplots(figsize=(12,8),dpi=200)
#zorder=[10,0];alpha=[1, 0.2] # show SDSS 
#zorder=[0,10];alpha=[0.2, 1] # show BOSS 
zorder=[0,0];alpha=[0.2, 0.2]

plot_embedding(embedded_batch,embed_color,name=name,comment=comment,
               cbar_label=cbar_label, c_range=c_range,cmap=cmap, zorder=zorder, alpha=alpha,
               markers=["o","^","s"],color=colors, ax=ax,mute=True)
ax.set_title("Random joint targets (N=%d)"%n_joint)

#ax.text(6,2,text,fontsize=15)
ax.text(0.1*xlim[1],0.5*ylim[1],text,fontsize=15)
ax.set_xlim(xlim);ax.set_ylim(ylim)
#plt.tight_layout()
plt.savefig("[2D]%d-joint-targets.png"%(n_joint),dpi=200)