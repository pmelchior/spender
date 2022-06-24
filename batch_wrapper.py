#!/usr/bin/env python
# coding: utf-8
import os,sys,time
import numpy as np 
import astropy.io.fits as fits
import multiprocessing as mp
import pickle,torch
import humanize,psutil
from util import get_norm
from line_profiler import LineProfiler
from memory_profiler import profile, memory_usage

def mem_report():
    print("CPU RAM Free: " + humanize.naturalsize( psutil.virtual_memory().available ))
    return

def read_sdss_spectra(plate, mjd, fiberid): 
    flocal = os.path.join(data_dir, 'sdss-spectra/spec-%s-%i-%s.fits' % (str(plate).zfill(4), mjd, str(fiberid).zfill(4)))
    if not os.path.isfile(flocal): 
        flocal = os.path.join(data_dir, 'lite/%s/spec-%s-%i-%s.fits' % (str(plate).zfill(4),str(plate).zfill(4), mjd, str(fiberid).zfill(4)))
        if not os.path.isfile(flocal):raise ValueError
    hdulist = fits.open(flocal)
    
    header = hdulist[0].header
    data = hdulist[1].data
    logw = data['loglam']
    spec = data['flux']
    ivar = data['ivar'] 
    mask = data['and_mask'].astype(bool)
    ivar[mask] = 0.
    return logw, spec, ivar, mask

def save_batch(batch_copy,batch_name):
    with open("%s/%s"%(dynamic_dir,batch_name), 'wb') as f:
        pickle.dump(batch_copy,f)
    print("Saving to %s/%s.."%(dynamic_dir,batch_name))
    return

def prepare_batch(input_list):
    targets,wave,code,wh_start,Nspec = input_list
    # padded wavelength range
    max_len = len(wave)
    specs = np.zeros((Nspec, len(wave)), dtype=np.float32)
    weights = np.zeros((Nspec, len(wave)), dtype=np.float32)
    norms = np.zeros(Nspec, dtype=np.float32)
    zreds = np.zeros(Nspec, dtype=np.float32)
    zerrs = np.zeros(Nspec, dtype=np.float32)
    specid = []
    
    ta = time.time()
    
    interval = Nspec//5

    i = 0
    for j in range(len(targets[0])):
        plate, mjd, fiberid = [targets[k][j] for k in [0,1,2]]
        try: 
            x = read_sdss_spectra(plate, mjd, fiberid)
        except:# (IndexError, OSError): 
            print("Error!",plate, mjd, fiberid)
            continue 

        Z = targets[3][j]
        Z_ERR = targets[4][j]
        
        #print("i:",j,"j:",j,plate, mjd, fiberid, Z, Z_ERR)
        if np.sum(x[1]) == 0:continue
        if Z <= 0:continue
        if Z > 0.5:continue
        if np.max(np.abs(x[1]))>1e5:continue
        
        iw = int(np.around((x[0][0] - wave[0])/0.0001))

        if iw < 0: 
            print("x[0][0]:",x[0][0])
            continue

        endpoint = min(iw+len(x[1]),max_len)

        if iw+len(x[1])>max_len:
            print("wavemax:",max(x[0]))
            print("max_len:",max_len,"iw:",iw,
                  "len(x[1]):",len(x[1]))

        norm = get_norm(np.expand_dims(x[1], axis=0))
        w = x[2] * ~x[3] * (norm**2)[:,None]
        
        if norm<0: 
            print("negative norm, continue...")
            continue
        
        if (w<=0).all():
            print("All zero weight, continue...",plate, mjd, fiberid)
            continue

        w[w<1e-6] = 1e-6  
        specs[i,iw:endpoint] = x[1][:endpoint-iw]/norm
        weights[i,iw:endpoint] = w[0][:endpoint-iw]
        zreds[i] = Z
        zerrs[i] = Z_ERR
        norms[i] = norm
        specid.append("%d-%d-%d"%(plate,mjd,fiberid))

        i += 1
        
        # collected all samples
        if i==Nspec:break
        if debug and (i % interval) == 0:
            tb = time.time()

            t_seg = tb-ta
            remain_seg = (Nspec-i)/interval
            print("\n\n%d: remaining time = %.2f"%(i,remain_seg*t_seg))
            ta = tb
            if debug:mem_report()
    
    if i<0.9*Nspec:
        print("filling fraction too low... %.2f"%(i/Nspec))
        return
    # save
    rand_int = np.arange(len(specs))
    print("len(specs):",len(specs),#"specid:",specid,
          "rand_int",rand_int.max())
    
    specid = np.array(specid)
    batch_copy = [torch.tensor(specs[rand_int],device=device), 
                  torch.tensor(weights[rand_int],device=device),
                  torch.tensor(zreds[rand_int],device=device), 
                  specid[rand_int],norms[rand_int],
                  torch.tensor(zerrs[rand_int],device=device)]
    
    batch_name = "%s%s_%d.pkl"%("",code,wh_start)
    save_batch(batch_copy,batch_name)
    return

def wrap_batches(which,tag,k_range=[0,10],Nspec=512):
    headers = {0:"truncated-specobj.pkl",
               1:"boss_headers.pkl"}
    
    header_name = headers[which]
    if "joint" in tag:header_name = "joint_headers.pkl"
        
    f = open(header_name,"rb")
    targets = pickle.load(f)
    f.close()
    # all targets
    name = ["SDSS","BOSS"]
    id_names = ['PLATE','MJD','FIBERID',"Z","Z_ERR"]
    print("All targets:",len(targets))
    if "joint" in header_name:
        keys = ["%s_%s"%(k,name[which]) for k in id_names]
    else: keys = id_names
    
    #targets_slice = {}
    #for k in keys:targets_slice[k] = targets[k]
    targets_slice = []
    for k in keys:targets_slice.append(targets[k].data)
    
    lower,upper = LOGWAVE_RANGE[which]
    wave = np.arange(lower,upper, 0.0001).astype(np.float32) 

    start,end = k_range[0]*Nspec,k_range[1]*Nspec
    start_index = np.arange(start,end, Nspec)
    nbatch = len(start_index)
    
    print("nbatch:",nbatch,"start_index:",start_index)
    
    code = "%s%s"%(name[which],tag)
    
    input_list = []
    for k,index in enumerate(start_index):
        loc = range(index,index+3*Nspec)
        print("loc:",len(loc),loc[:10])
        batch_targets=[item[loc] for item in targets_slice]
        args = [batch_targets,wave,code,index,Nspec]
        input_list.append(args)
        #print("prepared batch %d/%d, time=%.2f"%(k,nbatch,tb-ta))
    
    pool = mp.Pool(5)
    result = pool.map(func=prepare_batch, iterable=input_list)
    pool.close()
    pool.join()
    return

debug = False
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
data_dir = '/scratch/gpfs/yanliang'
dynamic_dir = "/scratch/gpfs/yanliang/dynamic-data"
LOGWAVE_RANGE = [[3.578, 3.97],[3.549, 4.0175]]

if "batch_wrapper" in sys.argv[0]:
    which = int(sys.argv[1])
    tag = sys.argv[2]
    wrap_batches(which,tag,k_range=[0,7])
    """
    lprofiler = LineProfiler()
    lprofiler.add_function(read_sdss_spectra)
    lp_wrapper = lprofiler(prepare_batch)
    lp_wrapper()
    lprofiler.print_stats()
    """