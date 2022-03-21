#!/usr/bin/env python
# coding: utf-8
import os, sys, time
import requests, json
import numpy as np
from PIL import Image
 
img_dir = "/scratch/gpfs/yanliang/image"
explore_url = "http://skyserver.sdss.org/dr16/en/tools/explore/obj.aspx"

if "expl" in sys.argv:
    item = "541 51959 428"
    plate,mjd,fiber = item.split()
    params = {"plate":plate,"mjd":mjd,"fiber":fiber}

    explore = requests.get(explore_url,params=params)
    print("\n\nFetching %s"%explore.url)
    print("\n\nexplore panel:",explore.content)
    exit()

image_url = "http://skyserver.sdss.org/dr16/SkyServerWS/ImgCutout/getjpeg"

download_from = sys.argv[1]
with open(download_from) as f:
    url_dict = json.load(f)
    names = url_dict.keys()

ra_dec_file = "RA-DEC-TRUE.json"
with open(ra_dec_file) as f:
    data_dict = json.load(f)

#keys = objects[0].keys()
size = "128"
scale ="0.4" # 0.4 arcsec / pixel
saved = os.listdir(img_dir)

for name in names:
    obj = data_dict[name]
    ra, dec = obj['ra'],obj['dec']
    ra,dec = [str(i) for i in [ra,dec]]
    plate,mjd,fiber = [int(i) for i in name.split("-")]
    params = {"ra":ra,"dec":dec,"scale":scale,
              "height": size, "width": size,"opt":""}

    img_name = "%d-%d-%d.jpg"%(plate,mjd,fiber)
    if img_name in saved:
        print("%s exists! Skipped..."%img_name)
        continue
    print("Downloading object: %s..."%img_name)
    img_data = requests.get(image_url,params=params).content
    with open('%s/%s'%(img_dir,img_name), 'wb') as handler:
        handler.write(img_data)    
    time.sleep(1)


    
