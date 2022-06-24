#!/usr/bin/env python
# coding: utf-8
import os,sys,time,socket
import numpy as np 
import astropy.table as aTable
import astropy.io.fits as fits
import multiprocessing
import urllib.request
import requests,json
import matplotlib.pyplot as plt
import pickle
socket.setdefaulttimeout(5)

dat_dir = '/scratch/gpfs/yanliang/sdss-spectra/'
#save_dir = "joint_headers.pkl"
paths = ["/scratch/gpfs/yanliang/sdss-spectra",
         "/scratch/gpfs/yanliang/lite"]
sdss_url = "https://data.sdss.org/sas/dr16/sdss/spectro/redux/26/spectra/lite/"
boss_url = "https://data.sdss.org/sas/dr16/eboss/spectro/redux/v5_13_0/spectra/lite/"


bulk_url = {"SDSS":sdss_url,"BOSS":boss_url}
save_dir = {"SDSS":"truncated-specobj.pkl","BOSS":"boss_headers.pkl"}

def spec_url(plate, mjd, fiberid):
    url = '%s/spec-%s-%i-%s.fits' % \
        (str(plate).zfill(4), str(plate).zfill(4), 
         mjd, str(fiberid).zfill(4)) 
    return url

def check_exists(input_list,loud=False):
    print("input_list:",len(input_list))
    output_list = []

    for item in input_list:
        plate, mjd, fiberid, _ = item
        filename = spec_url(plate, mjd, fiberid)
        
        exists = False
        for path in paths:
            if "lite" in path:
                full_path = "%s/%s"%(path,filename)
            else:full_path = "%s/%s"%(path,os.path.basename(filename))
            if os.path.exists(full_path):
                exists=True
                break
        if exists: 
            if loud:print("%s exists! skipping"%filename)
        else: output_list.append(item)
    print("output_list:",len(output_list))
    return output_list
    
    
def download_spectra(param):
    plate, mjd, fiberid, label = param

    url = '%s/%s/spec-%s-%i-%s.fits' % \
    (bulk_url[label],str(plate).zfill(4), str(plate).zfill(4), 
     mjd, str(fiberid).zfill(4))  
    
    flocal = os.path.join(dat_dir, os.path.basename(url))
    if os.path.isfile(flocal): 
        #print("%s exists! Skipped..."%os.path.basename(url))
        return None
    print("Downloading..",url)

    try: urllib.request.urlretrieve(url, flocal)
    except (IndexError, OSError): 
        print("Failed!")
        return None
    return None

def download_bulk_spectra(input_list,label,                saveto=".",listname="test"):

    text = ""
    for item in input_list:
        plate, mjd, fiberid, _ = item
        text += "%s\n"%spec_url(plate, mjd, fiberid)
    
    filename = "%s/%s-speclist.txt"%(saveto,listname)
    f = open(filename,"w")
    f.writelines(text)
    f.close()
    
    print("Saved %d lines to %s"%(len(input_list),filename))
    retrieve_url = bulk_url[label]
    command = "wget -nv -r -nH --cut-dirs=7 -i %s -B %s"%(os.path.basename(filename),retrieve_url)
    print("Use the following command:\n",command)
    #os.system(command)
    return

def read_joint():
    spec_file = 'SDSS_BOSS_common_targets.fits'
    print("Reading from: %s"%spec_file)

    specobj = aTable.Table.read(spec_file)

    sdss_class = specobj["CLASS_SDSS"]
    boss_class = specobj["CLASS_BOSS"]

    is_galaxy = sdss_class=="GALAXY"
    specobj = specobj[is_galaxy]
    
    print("sdss galaxy:",len(specobj))
    
    f = open(save_dir,"wb")
    pickle.dump(specobj,f)
    f.close()

    print("saved to %s"%save_dir)
    return


def headers_cut():
    spec_file = '/scratch/gpfs/yanliang/specObj-dr16.fits'
    print("Reading from: %s"%spec_file)

    specobj = aTable.Table.read(spec_file)
    keep = ((specobj['SURVEY'] == 'sdss  ') & # SDSS survey
            (specobj['PLATEQUALITY'] == 'good    ') & 
            (specobj['TARGETTYPE'] == 'SCIENCE ') 
           )
    print("keep:",np.sum(keep))
    zcut = ((specobj['Z'] > 0.) & (specobj['Z_ERR'] < 1e-4))
    print("zcut:",np.sum(keep & zcut) )
    is_galaxy = ((specobj['SOURCETYPE'] == 'GALAXY                   ') &
                 (specobj['CLASS'] == 'GALAXY'))
    print("galaxy:",np.sum(keep & zcut & is_galaxy))
    specobj = specobj[keep & zcut & is_galaxy]
    
    f = open(save_dir,"wb")
    pickle.dump(specobj,f)
    f.close()

    print("saved to %s"%save_dir)
    # just downloads spectra!!!
    return

id_names = ['PLATE','MJD','FIBERID']

label = sys.argv[1]
begin = 150000
batch_size = 50000


f = open(save_dir[label],"rb")
targets = pickle.load(f)
f.close()

tic = time.time()
end = min(begin+batch_size,len(targets))

#labels = ["SDSS","BOSS"]

input_list = []
for j in range(begin,end):
    plate, mjd, fiber = [targets[k][j] for k in id_names]
    input_list.append([plate, mjd, fiber, label])

output_list = check_exists(input_list)

if output_list == []:
    print("All spectra exist!")
    exit()

toc = time.time()
print('Done in {:.4f} seconds'.format(toc-tic))

# starts download...    
if "bulk" in sys.argv:
    download_bulk_spectra(output_list,label,saveto=save_dir[label],
                          listname="boss-500k")
    exit()
        
pool = multiprocessing.Pool(15)
result = pool.map(func=download_spectra, iterable=output_list)#, chunksize=n)
pool.close()
pool.join()
toc = time.time()
print('Done in {:.4f} seconds'.format(toc-tic))

    