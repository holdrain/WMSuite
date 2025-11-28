import os
import torch
import random
import torchvision.utils as vutils
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from wm.core.dataset import CustomImageFolder
from wm.core.helpers import (
    generate_random_fingerprints,
    msg2str,
    new_dir,
    str2msg,
    set_seeds,
)
from wm.core.yml import load_config
from wm.core.helpers import message_dict, transforms_dict_encode, load_prompts, image_resolution_dict
from wm.core.model_choice import (
    get_DwtDct,
    get_hiddenmodel,
    get_rivagan,
    get_stablesignature,
    get_Stegastamp,
    get_vine,
)


def run_stable_signature(opt,device):
    prompts_list, negative_prompts = load_prompts("wm/algorithms/config/stable_signature/prompt.yml")
    message = message_dict[opt.method]
    print(prompts_list,negative_prompts)

    cl_dir = new_dir(os.path.join(opt.output_dir,opt.method,'clean'))
    wm_dir = new_dir(os.path.join(opt.output_dir,opt.method,message))
    
    encoder_nowm,_ = get_stablesignature(device,nowm=True)
    encoder,_ = get_stablesignature(device,nowm=False)
    
    count = 0
    while count < opt.num:
        prompts = random.choices(prompts_list,k=opt.batch_size)
        with tqdm(total = opt.num,desc=f"generating wm images by {opt.method}") as pbar:
            set_seeds(count)
            clean_imgs = encoder_nowm(prompts,negative_prompt=negative_prompts * opt.batch_size,size=image_resolution_dict[opt.method])
            set_seeds(count)
            encoded_imgs = encoder(prompts,negative_prompt=negative_prompts * opt.batch_size,size=image_resolution_dict[opt.method])
            for clpil,encpil in zip(clean_imgs,encoded_imgs):
                enimg_path = os.path.join(wm_dir,f"{count:04d}"+opt.filetype)
                encpil.save(enimg_path)
                climg_path = os.path.join(cl_dir,f"{count:04d}"+opt.filetype)
                clpil.save(climg_path)
                count += 1
                pbar.update(1)


def run_hidden(opt,device,message='default'):
    cfgpath = "wm/algorithms/config/hidden/hidden.yaml"
    cfg = load_config(cfgpath)
    encoder,_ = get_hiddenmodel(cfg.train,device)

    ds = CustomImageFolder(opt.dataset,transform=transforms_dict_encode[opt.method],num=opt.num)
    dl = DataLoader(ds, batch_size = opt.batch_size, num_workers=0)

    cl_dir = new_dir(os.path.join(opt.output_dir,opt.method,'clean'))
    wm_dir = new_dir(os.path.join(opt.output_dir,opt.method,message))


    with tqdm(total=ds.__len__(),desc=f"generating wm images by {opt.method}") as pbar:
        count = 0
        for image in dl:
            if message == 'random':
                message = msg2str(generate_random_fingerprints(len(message_dict[opt.method])))
            else:
                message = message_dict[opt.method]

            image = image.to(device)
            message_tensor = torch.tensor(str2msg(message)).repeat(image.shape[0],1).to(device)
            encoded_img = encoder(image,message_tensor)
            for idx in range(encoded_img.shape[0]):
                vutils.save_image(encoded_img[idx],os.path.join(wm_dir,f"{count:04d}"+opt.filetype),normalize=True)
                vutils.save_image(image[idx],os.path.join(cl_dir,f"{count:04d}"+opt.filetype),normalize=True)
                count += 1
                pbar.update(1)



def run_rivagan(opt,message='default'):
    ds = CustomImageFolder(opt.dataset,transform=transforms_dict_encode[opt.method],num=opt.num)
    dl = DataLoader(ds, batch_size = 1,shuffle=False, num_workers=0)

    cl_dir = new_dir(os.path.join(opt.output_dir,opt.method,'clean'))
    wm_dir = new_dir(os.path.join(opt.output_dir,opt.method,message))

    with tqdm(total=ds.__len__(),desc=f"generating wm images by {opt.method}") as pbar:
        count = 0
        for image in dl:
            if message == 'random':
                message = msg2str(generate_random_fingerprints(len(message_dict[opt.method])))
            else:
                message = message_dict[opt.method] 
            
            encoder,_ = get_rivagan(wm_text=message)
            encoded_img = encoder(image)
            vutils.save_image(encoded_img,os.path.join(wm_dir,f"{count:04d}"+opt.filetype),normalize=True)
            vutils.save_image(image,os.path.join(cl_dir,f"{count:04d}"+opt.filetype),normalize=True)
            count += 1
            pbar.update(1)
    

def run_dwtdct(opt,message='default'):
    ds = CustomImageFolder(opt.dataset,transform=transforms_dict_encode[opt.method],num=opt.num)
    dl = DataLoader(ds, batch_size = 1, shuffle=False, num_workers=0)

    cl_dir = new_dir(os.path.join(opt.output_dir,opt.method,'clean'))
    wm_dir = new_dir(os.path.join(opt.output_dir,opt.method,message))

    with tqdm(total=ds.__len__(),desc=f"generating wm images by {opt.method}") as pbar:
        count = 0
        for image in dl:
            if message == 'random':
                message = msg2str(generate_random_fingerprints(len(message_dict[opt.method])))
            else:
                message = message_dict[opt.method]
            
            encoder,_ = get_DwtDct(wm_text=message,wm_type='bits')
            encoded_img = encoder(image)
            vutils.save_image(encoded_img,os.path.join(wm_dir,f"{count:04d}"+opt.filetype),normalize=True)
            vutils.save_image(image,os.path.join(cl_dir,f"{count:04d}"+opt.filetype),normalize=True)
            count += 1
            pbar.update(1)


def run_stegastamp(opt,device,message='default'):
    encoder,decoder = get_Stegastamp(device)

    ds = CustomImageFolder(opt.dataset,transform=transforms_dict_encode[opt.method],num=opt.num)
    dl = DataLoader(ds, batch_size = opt.batch_size, shuffle=False, num_workers=0)
    
    cl_dir = new_dir(os.path.join(opt.output_dir,opt.method,'clean'))
    wm_dir = new_dir(os.path.join(opt.output_dir,opt.method,message))

    with tqdm(total=ds.__len__(),desc=f"generating wm images by {opt.method}") as pbar:
        count = 0
        for image in dl:
            if message == 'random':
                message = msg2str(generate_random_fingerprints(len(message_dict[opt.method])))
            else:
                message = message_dict[opt.method]

            image = image.to(device)
            message_tensor = torch.tensor(str2msg(message)).repeat(image.shape[0],1).to(device)
            encoded_img = encoder(message_tensor,image)

            for idx in range(encoded_img.shape[0]):
                vutils.save_image(encoded_img[idx],os.path.join(wm_dir,f"{count:04d}"+opt.filetype),normalize=True)
                vutils.save_image(image[idx],os.path.join(cl_dir,f"{count:04d}"+opt.filetype),normalize=True)
                count += 1
                pbar.update(1)


def run_vine(opt,device,message='default'):
    encoder,_ = get_vine(device)
    ds = CustomImageFolder(opt.dataset,transform=transforms_dict_encode[opt.method],num=opt.num)
    dl = DataLoader(ds, batch_size = 1, shuffle=False, num_workers=0)

    cl_dir = new_dir(os.path.join(opt.output_dir,opt.method,'clean'))
    wm_dir = new_dir(os.path.join(opt.output_dir,opt.method,message))
    
    with tqdm(total=ds.__len__(),desc=f"generating wm images by {opt.method}") as pbar:
        count = 0
        for image in dl:
            if message == 'random':
                message = msg2str(generate_random_fingerprints(len(message_dict[opt.method])))
            else:
                message = message_dict[opt.method] 
            
            image = image.to(device)
            message_tensor = torch.tensor(str2msg(message)).repeat(image.shape[0],1).to(device)
            encoded_img = encoder(image,message_tensor)

            for idx in range(encoded_img.shape[0]):
                vutils.save_image(encoded_img[idx],os.path.join(wm_dir,f"{count:04d}"+opt.filetype),normalize=True)
                vutils.save_image(image[idx],os.path.join(cl_dir,f"{count:04d}"+opt.filetype),normalize=True)
                count += 1
                pbar.update(1)
