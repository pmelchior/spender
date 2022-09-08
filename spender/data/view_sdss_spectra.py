#!/usr/bin/env python
# coding: utf-8
import io, os, sys, time, random
sys.path.insert(1, './')
import numpy as np
import pickle
import torch
from torchinterp1d import Interp1d
from spender import SpectrumAutoencoder
from spender.data.sdss import SDSS, BOSS
from spender.util import mem_report, resample_to_restframe
from spender.data.emission_lines import emissionlines
import humanize,psutil,GPUtil
import matplotlib.pyplot as plt

data_dir = "/scratch/gpfs/yanliang"
dynamic_dir = "/scratch/gpfs/yanliang/dynamic-data"
savemodel = "models"
device =  torch.device("cpu")

# restframe wavelength for reconstructed spectra
# Note: represents joint dataset wavelength range
lmbda_min = 2359#data['wave'].min()/(1+data['z'].max())
lmbda_max = 10402#data['wave'].max()
bins = 7000#int(data['wave'].shape[0] * (1 + data['z'].max()))
wave_rest = torch.linspace(lmbda_min, lmbda_max, bins, dtype=torch.float32)
print ("Restframe:\t{:.0f} .. {:.0f} A ({} bins)".format(lmbda_min, lmbda_max, bins))

MY_COLORS = ['mediumseagreen',"skyblue",'salmon','orange',
             'gold',"royalblue","yellow","deeppink","deepskyblue",
             'orangered','aqua']

def permute_indices(length,n_redundant=1):
    wrap_indices = torch.arange(length).repeat(n_redundant)
    rand_permut = wrap_indices[torch.randperm(length*n_redundant)]
    return rand_permut


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

def load_model(path, instruments, wave_rest,n_latent=2, normalize=True):
    device = wave_rest.device
    mdict = torch.load(path, map_location=device)

    models = []
    for j in range(len(instruments)):
        model = SpectrumAutoencoder(instruments[j],
                                    wave_rest,
                                    n_latent=n_latent,
                                    normalize=normalize)
        model.load_state_dict(mdict["model"][j])
        models.append(model)
    return models,mdict["losses"]

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

def spectra_multiplot(params, instruments, which, offset=3,
                locid=[], prim_xlim=[],plot=True,axs_list=None):
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
            wrest = inspect_mix[j].decoder.wave_rest
            wave_data = waves[j]/(1+true_z[j])
            locwave = waves[j][None,:]

            batch = (ydata[j],weights[j],true_z[j])
            recon_ij,fake_w,tensor_znew=instruments[j].augment_spectra(batch,noise=False)
            z_new = tensor_znew.item()
            print("z=%.2f weight mean:%.2f weight std:%.2f"%(z_new,fake_w.mean(),fake_w.std()))
            print("recon_ij: %.2f +/= %.2f"%(recon_ij.mean(),
                  recon_ij.std()))
            
            # model arguments
            args = (recon_ij.float(), fake_w.float(), instruments[j],
                    tensor_znew)
            latent,y_rest,_ = inspect_mix[j]._forward(*args)
            
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

            label = "z=%.2f (loss=%.2f)"%(z_new,loss)
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
        
        latent,y_rest,_ = inspect_mix[j]._forward(*args)
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

        s_sim,spec_sim,sim_loss = similarity_restframe(\
        instruments[j],inspect_mix[j],embed[j],individual=True)
        print("s_chi:",s_sim[-1])
        print("spec_chi:",spec_sim[-1])
        print("similarity loss:",sim_loss[-1])
    

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

    cbar = plt.colorbar(img,ax=ax,location="right",fraction=0.05,
                        shrink=0.8,pad=0.01)
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
            
    ax.legend(loc='best')
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

def similarity_loss(instrument, model, spec, w, z, s, slope=1.0, individual=False, wid=5):
    spec,w = resample_to_restframe(instrument.wave_obs,
                                   model.decoder.wave_rest,
                                   spec,w,z)

    batch_size, spec_size = spec.shape
    _, s_size = s.shape
    device = s.device

    # pairwise dissimilarity of spectra
    S = (spec[None,:,:] - spec[:,None,:])**2

    # pairwise weights
    non_zero = w > 0
    N = (non_zero[None,:,:] * non_zero[:,None,:])
    W = (1 / w)[None,:,:] + (1 / w)[:,None,:]
    W =  N / W

    N = N.sum(-1)
    N[N==0] = 1 
    # dissimilarity of spectra
    # of order unity, larger for spectrum pairs with more comparable bins
    spec_sim = (W * S).sum(-1) / N

    # dissimilarity of latents
    s_sim = ((s[None,:,:] - s[:,None,:])**2).sum(-1) / s_size

    # only give large loss of (dis)similarities are different (either way)
    x = s_sim-spec_sim
    sim_loss = torch.sigmoid(slope*x-wid/2)+torch.sigmoid(-slope*x-wid/2)
    diag_mask = torch.diag(torch.ones(batch_size,device=device,dtype=bool))
    sim_loss[diag_mask] = 0

    if individual:
        return s_sim,spec_sim,sim_loss

    # total loss: sum over N^2 terms,
    # needs to have amplitude of N terms to compare to fidelity loss
    return sim_loss.sum() / batch_size

def restframe_weight(model,instrument,mu=[5000,5000],
                     sigma=[1000,1000],amp=[10,10]):
    if type(instrument).__name__ == "SDSS": i=0
    if type(instrument).__name__ == "BOSS": i=1
    x = model.decoder.wave_rest
    return amp[i]*torch.exp(-(0.5*(x-mu[i])/sigma[i])**2)

def similarity_restframe(instrument, model, s, slope=0.5,
                         individual=False, wid=5):
    _, s_size = s.shape
    device = s.device

    spec = model.decode(s)
    spec /= spec.median(dim=1)[0][:,None]
    batch_size, spec_size = spec.shape
    # pairwise dissimilarity of spectra
    S = (spec[None,:,:] - spec[:,None,:])**2
    # dissimilarity of spectra
    # of order unity, larger for spectrum pairs with more comparable bins
    W = restframe_weight(model,instrument)
    print("\n\nW:",W.mean())
    spec_sim = (W * S).sum(-1) / spec_size
    # dissimilarity of latents
    s_sim = ((s[None,:,:] - s[:,None,:])**2).sum(-1) / s_size

    # only give large loss of (dis)similarities are different (either way)
    x = s_sim-spec_sim
    sim_loss = torch.sigmoid(slope*x-wid/2)+torch.sigmoid(-slope*x-wid/2)
    diag_mask = torch.diag(torch.ones(batch_size,device=device,dtype=bool))
    sim_loss[diag_mask] = 0

    if individual:
        return s_sim,spec_sim,sim_loss

    # total loss: sum over N^2 terms,
    # needs to have amplitude of N terms to compare to fidelity loss
    return sim_loss.sum() / batch_size

def plot_annealing_schedule(slope=[0,1], wid=5, cycle=10, xrange=(-50,20),
                            cmap="inferno",lw=2):
    k1 = np.arange(slope[0],slope[1], 1.0/cycle)
    x = torch.arange(xrange[0],xrange[1],0.01)
    get_cmap = matplotlib.cm.get_cmap(cmap)
    fig,ax=plt.subplots(figsize=(8,5),constrained_layout=True)
    for i in range(cycle):
        #loss = torch.sigmoid(x)+torch.sigmoid(-k1[i]*x-wid)
        loss = torch.sigmoid(k1[i]*x-wid/2)+torch.sigmoid(-k1[i]*x-wid/2)
        ax.plot(x,loss,lw=lw,c=get_cmap(i/cycle))
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=0, vmax=1))
    cbar = fig.colorbar(sm)
    cbar.set_label("$k_1$",fontsize=15)
    ax.set_xlabel("$(\Delta s)^2-(\Delta y)^2$")
    ax.set_ylabel("loss")
    plt.savefig("annealing.png",dpi=200)
    return

def plot_detailedloss(loss, ax=None, fs=12):   
    loss = np.array(loss[1])
    mask = (loss[0].T[0]>0)
    print("mask:",mask.shape)
    epoch = mask.sum()
    ep = np.arange(epoch)
    print("epoch:",epoch)
    if not ax:fig,ax=plt.subplots(figsize=(8,6),dpi=200)
    name = ["S","B"]
    labels = ["fidelity","similarity",
              "aug_fidelity","aug_similarity","consistency"]
    colors = [['k','r','k','r',"c"],["b","orange","b","orange","pink"]]
    style = ["-","-","--","--","-"]
    alphas = [1,1,1,1,1]
    lws = [2,2,2,2,2]
    for j,nn in enumerate(name):
        locloss = loss[j][mask].T
        print("locloss:", locloss.shape)
        for i,ll in enumerate(labels):
            final_ep = locloss[i][-1]
            if sum(locloss[i])==0:continue
            ax.semilogy(ep,locloss[i],style[i],
                        label="%s %s(%.2f)"%
                        (ll,nn,final_ep),lw=lws[i],
                        alpha=alphas[i],color=colors[j][i])
        
    #ax.axhline(0.5,ls="--",color="b",label="loss = 0.5")
    ax.set_xlabel("epoch",fontsize=fs)
    #ax.set_ylabel("loss",fontsize=fs)
    ax.legend(loc="best",fontsize=fs,ncol=2)
    ax.set_title("Validation Losses",fontsize=fs)
    #plt.savefig("detailed_loss.png",dpi=200)
    return

def plot_similarity(model, data, fs=15, n_sim = 100, ax=None):
    specs,weights,redshifts,instruments,names = data
    rands = [np.random.randint(len(redshifts[0]), size=n_sim),
             np.random.randint(len(redshifts[1]), size=n_sim)]
    
    losses = []; colors = ["salmon","skyblue"];cmap = "rainbow"
    if not ax:
        fig,ax = plt.subplots(figsize=(8,5),constrained_layout=True)
    
    for j in range(len(instruments)):
        rand = rands[j]

        s = model[j].encode(specs[j][rand],
                            aux=redshifts[j][rand].unsqueeze(1))

        detail = similarity_restframe(instruments[j],model[j],s,
                                      individual=True)

        s_sim,spec_sim,sim_loss = [i.detach().cpu() for i in detail]
        print("\ns_sim = %.2f +/- %.2f"%(s_sim.median(),s_sim.std()))
        print("spec_sim = %.2f +/- %.2f\n"%(spec_sim.median(),spec_sim.std()))
        label_j = "%s(%.1e,%.1e)"%(names[j],s_mat[j][0].std(),s_mat[j][1].std())
        img = ax.scatter(spec_sim,s_sim,c=sim_loss, vmin=0, vmax=1,
                         cmap=cmap,edgecolors=colors[j],
                         label=label_j)
        losses.append(sim_loss.mean())

    comment = "similarity loss: %.3f, %.3f"%(losses[0],losses[1])
    cbar = plt.colorbar(img,ax=ax,location="right",fraction=0.05,
                        shrink=0.8, pad=0.01)
    cbar.set_label("similarity loss",fontsize=fs)
    ax.set_title(comment,fontsize=fs)
    ax.legend(loc="best",fontsize=fs)
    ax.set_ylabel("latent $\chi^2$",fontsize=fs)
    ax.set_xlabel("spectral $\chi^2$",fontsize=fs)
    return

def plot_joint_samples(s_mat, model, data, zorder=[0,0], alpha=[0.2, 0.2],
                       n_joint = 200, ax=None, xlim=None,ylim=None):
    specs,weights,redshifts,instruments,wh_joints = data
    rand = np.array(random.sample(range(len(wh_joints[0])), n_joint))
    s_track = []
    mark_ids = rand
    for j in range(n_encoder):
        inds = np.array(wh_joints[j])[rand]
        s = model[j].encode(specs[j][inds],
                            aux=redshifts[j][inds].unsqueeze(1))

        s = s.detach().numpy().T
        s_track.append(s)
    diff = np.sqrt(np.sum((s_track[0]-s_track[1])**2, axis=0))
    med_diff = np.median(diff)

    dist_rank = np.argsort(diff)[::-1]
    diff = diff[dist_rank]
    mask_names = mark_ids[dist_rank]

    text = "Rank   No.   Distance\n"
    for ii in range(5):
        text += "%d         %d       %.2f\n"%(ii+1,mask_names[ii],diff[ii])
    text += "(median distance = %.3f)"%(med_diff)
    colors = [['gold','navy']]
    
    embedded_batch = [s_mat,s_track]
    comment = [{"ID":mark_ids,"wh":0, "pos":(-10,0),"fs":5}]
    if not ax:fig,ax = plt.subplots(figsize=(12,8),dpi=200)
    plot_embedding(embedded_batch,embed_color,name= ["target","augment"],
                   comment=comment,
                   cbar_label=cbar_label, c_range=c_range,cmap=cmap, 
                   zorder=zorder, alpha=alpha,
                   markers=["o","^","s"],color=colors, ax=ax,mute=True)
    ax.set_title("Random joint targets (N=%d)"%n_joint)
    
    ax.text(xlim[0]+0.5*(xlim[1]-xlim[0]),ylim[0]+0.7*(ylim[1]-ylim[0]),text,fontsize=15)
    ax.set_xlim(xlim);ax.set_ylim(ylim)
    return

def plot_model(data, model, axs=[],fs=15):
    specs,weights,redshifts,instruments,wh_joints=data
    # visualize 10 latent space
    if axs==[]:
        fig,axs = plt.subplots(1,3,figsize=(8,4),dpi=200,
                               constrained_layout=True,
                               gridspec_kw={'width_ratios': [1.5, 1, 1]})
    col = loss[0][0].sum(axis=0)>0
    losses = np.zeros((4,loss.shape[2]))
    losses[0] = 2*loss[0][0].T[col].mean(axis=0)
    losses[1] = 2*loss[1][0].T[col].mean(axis=0)
    losses[2] = 2*loss[0][1].T[col].mean(axis=0)
    losses[3] = 2*loss[1][1].T[col].mean(axis=0)

    print("\n\n",model_file,"losses:",losses.shape)
    
    non_zero = losses[0]>0
    losses = losses[:,non_zero]
    plot_loss(losses,ax=axs[0],fs=fs)
    
    np.random.seed(23)#42 
    rand = np.random.randint(len(wh_joints[0]), size=10)
    for j in range(n_encoder):
        ax=axs[j+1]
        inds = np.array(wh_joints[j])[rand]
        zlist = redshifts[j][inds].detach().numpy()
        rank = np.argsort(zlist)
        inds = inds[rank]
        s = model[j].encode(specs[j][inds],
                            aux=redshifts[j][inds].unsqueeze(1))
        s = s.detach().numpy()
        batch_size,s_size = s.shape
        locid = ["z=%.2f(%d)"%(n,ii) for ii,n in zip(rand[rank],redshifts[j][inds])] 

        vmin,vmax=s.min(),s.max()
        print("smin,smax:",vmin,vmax)
        ax.matshow(s,vmax=vmax,vmin=vmin)#,cmap='winter')
        ax.set_xticks([]);ax.set_yticks([])
        ax.set_xlabel("latent variables");ax.set_ylabel(names[j])
        if j>0: continue
        for i in range(len(locid)):
            ax.text(2,i+0.2,locid[i],c="k", weight='bold', 
                    fontsize=12,alpha=1)
    axs[0].set_title("(%s) best loss = %.2f"%(model_file,min(losses[1])),fontsize=fs)
    return

def compare_spectra(model,instrument,s):
    xx = model.decoder.wave_rest
    #spec_loc = model.decode(s).detach().numpy()
    #spec_loc /= np.median(spec_loc,axis=1)[:,None]
    rand = np.random.randint(len(ids_boss), size=10)
    spec_loc, wloc = resample_to_restframe(wave_boss,\
                     xx,spec_boss[rand],w_boss[rand],z_boss[rand])
    spec_loc = spec_loc.detach().numpy()

    wfunc = restframe_weight(model,instrument,amp=[1,1])
    plt.figure(figsize=(10,8))
    for i,yy in enumerate(spec_loc):
        plt.plot(xx,yy,c=MY_COLORS[i],label="spectrum %d"%i)
    plt.plot(xx,wfunc,"k--",lw=5,label="weight function")
    plt.legend()
    plt.xlabel("wave_rest $(\AA)$")
    plt.ylim(0,3)
    plt.savefig("compare_spectra.png",dpi=200)
    #count_rate(inspect_mix[0],s_sdss)
    exit()
    return
#-------------------------------------------------------
import matplotlib
plot_annealing_schedule()

instrument_names = ["SDSS", "BOSS"]
instruments = [ SDSS(), BOSS() ]
ins_sdss,ins_boss = instruments
n_encoder = len(instruments)

option_normalize = True

model_file = sys.argv[1]
inspect_mix, loss = load_model("%s"%(model_file),
                               instruments,wave_rest,n_latent=2)
[model.eval() for model in inspect_mix]

epoch = max(np.nonzero(loss[0][0])[0])+1
print("loss:",loss.shape,"epoch:",epoch)
print("Train SDSS :", loss[0][0][epoch-3:epoch])
print("Train BOSS :", loss[0][1][epoch-3:epoch])
print("Valid SDSS :", loss[1][0][epoch-3:epoch])
print("Valid BOSS :", loss[1][1][epoch-3:epoch])

random_seed = 0
random.seed(random_seed)

testset = collect_batches("chunk1024",which="test",NBATCH=25)
view = collect_batches("joint256",which=None)#,which="test")
view[0] += testset[0]
view[1] += testset[1]

wave_sdss = instruments[0].wave_obs
wave_boss = instruments[1].wave_obs

spec_sdss,w_sdss,z_sdss,ids_sdss,norm_sdss = merge_batch(view[0])
spec_boss,w_boss,z_boss,ids_boss,norm_boss = merge_batch(view[1])
w_sdss[:, instruments[0].skyline_mask] = 0
w_boss[:, instruments[1].skyline_mask] = 0

SDSS_id, BOSS_id =  boss_sdss_id()
joint_ids = find_intersection(ids_sdss,ids_boss)
wh_joint_sdss = [ids_sdss.index(i) for i in joint_ids]
wh_joint_boss = [ids_boss.index(i) for i in sdss2boss(joint_ids)]

names = ["SDSS spectra", "BOSS spectra"]
specs = [spec_sdss,spec_boss]
weights = [w_sdss, w_boss]
redshifts = [z_sdss, z_boss]
lognorms = [np.log10(norm_sdss), np.log10(norm_boss)]
wh_joints = [wh_joint_sdss,wh_joint_boss]
interesting = [511,595]#,244,480,511,542,1093]

if "model" in sys.argv:
    data = [specs,weights,redshifts,instruments,wh_joints]
    plot_model(data, inspect_mix, axs=[])
    plt.savefig("check_model.png",dpi=200)
if "alone" in sys.argv:exit()
    
# background points
np.random.seed(42)
N = 2000
rand1 = np.random.randint(len(ids_sdss), size=N)
rand2 = np.random.randint(len(ids_boss), size=N)

s_sdss = inspect_mix[0].encode(spec_sdss[rand1], aux=z_sdss[rand1].unsqueeze(1))
s_boss = inspect_mix[1].encode(spec_boss[rand2], aux=z_boss[rand2].unsqueeze(1))

if "count" in sys.argv:
    compare_spectra(inspect_mix[1],instruments[1],s_boss[:10])

print("[Random sample] SDSS:",s_sdss.shape)#,"BOSS:",s_boss.shape)
vmin,vmax=s_sdss.min().detach(),s_sdss.max().detach()
s_size = s_sdss.shape[1]

embed_color=[z_sdss[rand1],z_boss[rand2]]
c_range = [0,0.5];cmap="gist_rainbow";cbar_label = "$z$"

s_mat= [s_sdss.detach().numpy().T,s_boss.detach().numpy().T]
s=np.hstack((s_mat[0],s_mat[1]))
print("\n\ns:",s.shape)
xlim  = (s[0].min(),s[0].max())
ylim  = (s[1].min(),s[1].max())

fig = plt.figure(constrained_layout=True,figsize=(18,12),dpi=200)
gs = fig.add_gridspec(2, 6, width_ratios=[3,1,1,1,1,1],hspace=0.1)

ax11 = fig.add_subplot(gs[0, 0])
ax12 = fig.add_subplot(gs[0, 1])
ax13 = fig.add_subplot(gs[0, 2])
ax1 = [ax11,ax12,ax13]

ax2 = fig.add_subplot(gs[0, 3:])
ax3 = fig.add_subplot(gs[1, :2])
ax4 = fig.add_subplot(gs[1, 2:])

plot_detailedloss(loss,ax=ax2)

# plot similarity
plot_similarity(inspect_mix, [specs,weights,redshifts,
                              instruments,names],ax=ax3)

data = [specs,weights,redshifts,instruments,wh_joints]
plot_model(data, inspect_mix, axs=ax1)
# plot joint samples
plot_joint_samples(s_mat,inspect_mix, data,ax=ax4,xlim=xlim,ylim=ylim)
plt.savefig("[2D]joint-targets.png",dpi=200)
#exit()

# define augmented samples 
nparam = 3
use_data=[];number = np.zeros(nparam);
newz = np.array([0.1,0.3,0.5])

#newz = np.array([-1e-2,1e-2])
#number = np.array([0,0])
locid = ["z=%.1f"%(zz) for zz,n in zip(newz,number)]
locid += ["raw data"]
newz[newz<0]=0;newz[newz>0.5]=0.5
params = np.array([newz,number]).T


name = ["target","augment"]
colors = [['salmon','skyblue'],['salmon','skyblue'],['orange',"navy"],['gold','azure']]
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
    spectra_multiplot(params,instruments,which,
                locid=locid,offset=10,
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

    mark_id = np.array(ids_sdss)[wh_sdss]
    comment = [{"ID":mark_id,"wh":1},
               {"pos":(-5,0),"ID":locid+[None],"wh":1}]
    plot_embedding(embedded_batch,embed_color,name=name,comment=comment,
                   cbar_label=cbar_label, c_range=c_range,
                   markers=["o","^","s"],color=colors, ax=ax9, mute=True, cmap=cmap)
    
    ax9.set_xlim(xlim);ax9.set_ylim(ylim)

    text = "Data similarity:\nSDSS/BOSS as model: %.2f vs. %.2f\n"%(cross_loss[1],cross_loss[0])
    text += "SDSS/BOSS weight: %.2f vs. %.2f"%(mean_weights[0],mean_weights[1])
    ax9.text(xlim[0],0.8*ylim[1],text,fontsize=20)

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

