#!/usr/bin/env python
# coding: utf-8
import os, sys, time
import requests, json
import numpy as np
from PIL import Image
 
img_dir = "image"
merge_dir = "outliers"
explore_url = "http://skyserver.sdss.org/dr16/en/tools/explore/obj.aspx"



if "merge" in sys.argv:
    imfiles = os.listdir(merge_dir)
    
    imfiles = [f for f in imfiles if ".png" in f]
    imfiles = sorted(imfiles)
    #print(imfiles)
    
    im_list = []
    for filename in imfiles:
        im = Image.open("%s/%s"%(merge_dir,filename))
        im.load() # required for png.split()

        copy = Image.new("RGB", im.size, (255, 255, 255))
        copy.paste(im, mask=im.split()[3]) # 3 is the alpha channel
        
        im_list.append(copy)
    
    n = len(imfiles)*5
    pdf_name = "top-%d-spectra.pdf"%n

    im1=im_list[0]
    print("Saveing %d pages to %s"%(len(im_list),pdf_name))
    im1.save(pdf_name, "PDF" ,resolution=100.0, 
             save_all=True, append_images=im_list[1:])
    exit()

if "expl" in sys.argv:
    item = "408 51821 208"
    plate,mjd,fiber = item.split()
    params = {"plate":plate,"mjd":mjd,"fiber":fiber}

    explore = requests.get(explore_url,params=params)
    print("\n\nFetching %s"%explore.url)
    print("\n\nexplore panel:",explore.content)
    exit()

image_url = "http://skyserver.sdss.org/dr16/SkyServerWS/ImgCutout/getjpeg"

with open('sdss-top50.json') as f:
    all_data,query = json.load(f)
objects = all_data['Rows']
#keys = objects[0].keys()
size = 128
scale ="0.4" # 0.4 arcsec / pixel

for obj in objects:
    ra, dec = obj['ra'],obj['dec']
    plate,mjd,fiber = obj['plate'],obj['mjd'],obj['fiberid']

    ra,dec,size = [str(i) for i in [ra,dec,size]]
    params = {"ra":ra,"dec":dec,"scale":scale,
              "height": size, "width": size,"opt":""}

    img_name = "%d-%d-%d"%(plate,mjd,fiber)
    
    print("Downloading object: %s..."%img_name)
    img_data = requests.get(image_url,params=params).content
    with open('%s/%s.jpg'%(img_dir,img_name), 'wb') as handler:
        handler.write(img_data)    
    time.sleep(1)
    #exit()

    
