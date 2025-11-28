import argparse
import os
import torch
from tqdm.auto import tqdm
from datetime import datetime

from wm.core.dataset import CustomImageFolder
from wm.core.helpers import msg2str, str2msg
from wm.core.yml import load_config
from wm.core.helpers import transforms_dict_decode,cal_tolerant,message_length_dict,message_dict,set_seeds
from wm.core.model_choice import (
    get_DwtDct,
    get_hiddenmodel,
    get_rivagan,
    get_stablesignature,
    get_vine,
    get_Stegastamp,
)


def Options():
    parser = argparse.ArgumentParser(description="")
    parser.add_argument('--data_path', type=str, help='dir used to embed watermarks')
    parser.add_argument('--method', type=str, default='rivaGan', choices=['dwtdct','stegastamp', 'hidden', 'rivaGan','stable_signature','vine'], help='watermark algorithms')
    parser.add_argument('--device', type=str, default='cuda:0', help='device to use')
    parser.add_argument('--beta', type=float, default=0.000001, help='beta for cal_tolerant')
    parser.add_argument('--log_dir',type=str, help='dir to save the decoding results file')
    args = parser.parse_args()
    return args

def load_models(method, device):
    if method == 'dwtdct':
        return get_DwtDct(wm_text=message_dict[method], wm_type='bits')
    elif method == 'hidden':
        cfgpath = "wm/algorithms/config/hidden/hidden.yaml"
        cfg = load_config(cfgpath)
        return get_hiddenmodel(cfg.train, device)
    elif method == 'stegastamp':
        return get_Stegastamp(device)
    elif method == 'rivaGan':
        return get_rivagan(wm_text=message_dict[method])
    elif method == 'stable_signature':
        return get_stablesignature(device)
    elif method == 'vine':
        return get_vine()
    else:
        raise ValueError(f"Unsupported method: {method}")

def extract_watermark(opt,decoder):
    ds = CustomImageFolder(opt.data_path, transform=transforms_dict_decode[opt.method])
    image_files = ds.filenames
    ts = datetime.now().strftime("%Y%m%d_%H%M%S") 
    output_file = os.path.join(opt.log_dir, f"results_{ts}.txt")
    tolerant_bits = cal_tolerant(message_length_dict[opt.method],beta=opt.beta)
    if opt.method == 'dwtdct' or opt.method == 'rivaGan':
        opt.device = 'cpu'
    message = message_dict[opt.method]
    message_tensor = torch.Tensor(str2msg(message)).to(opt.device)

    bitwise_accuracy_sum = 0.0
    tp_sum = 0

    with tqdm(initial=0, total=len(image_files), desc="decoding watermark from images...") as pbar:
        with open(output_file, 'a') as f:
            f.write("Path\ttarge message\tdecoded message\tbitacc\n")
            for idx in range(len(image_files)):
                image = ds.__getitem__(idx)
                image = image.unsqueeze(0).to(opt.device)
                decoded_message = decoder(image)
                decoded_message = decoded_message.round().clip(0, 1).long()
                difference = (decoded_message != message_tensor).float()
                if difference.sum().item() <= tolerant_bits:
                    tp_sum += 1
                bitwise_accuracy = (1.0 - difference.mean()).item()
                bitwise_accuracy_sum += bitwise_accuracy
                
                f.write(f"{image_files[idx]}\t{message}\t{msg2str(decoded_message)}\t{bitwise_accuracy}\n")
                pbar.update(1)
            bitwise_accuracy_avg = bitwise_accuracy_sum / len(image_files)
            TPR = tp_sum / len(image_files)
            f.write(f"bitwise_accuracy:{round(bitwise_accuracy_avg, 4)}\n")
            f.write(f"TPR:{round(TPR, 4)}\n")
    return bitwise_accuracy_avg,TPR


if __name__ == '__main__':
    set_seeds(2024)
    opt = Options()
    _, decoder = load_models(opt.method, opt.device)
    bitwise_accuracy_avg,FPR = extract_watermark(opt,decoder)
    print(round(bitwise_accuracy_avg, 4),round(FPR, 4))
