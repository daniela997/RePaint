# Copyright (c) 2022 Huawei Technologies Co., Ltd.
# Licensed under CC BY-NC-SA 4.0 (Attribution-NonCommercial-ShareAlike 4.0 International) (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode
#
# The code is released for academic research use only. For commercial use, please contact Huawei Technologies Co., Ltd.
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# This repository was forked from https://github.com/openai/guided-diffusion, which is under the MIT license

"""
Like image_sample.py, but use a noisy image classifier to guide the sampling
process towards more realistic images.
"""

import os
import argparse
import torch as th
import torch
import torch.nn.functional as F
import time
import conf_mgt
from utils import yamlread
from guided_diffusion import dist_util

# Workaround
try:
    import ctypes
    libgcc_s = ctypes.CDLL('libgcc_s.so.1')
except:
    pass


from guided_diffusion.script_util import (
    NUM_CLASSES,
    model_and_diffusion_defaults,
    classifier_defaults,
    create_model_and_diffusion,
    create_classifier,
    select_args,
)  # noqa: E402

def toU8(sample):
    if sample is None:
        return sample

    sample = ((sample + 1) * 127.5).clamp(0, 255).to(th.uint8)
    sample = sample.permute(0, 2, 3, 1)
    sample = sample.contiguous()
    sample = sample.detach().cpu().numpy()
    return sample


def main(conf: conf_mgt.Default_Conf):

    print("Start", conf['name'])

    device = dist_util.dev(conf.get('device'))


    model, diffusion = create_model_and_diffusion(
        **select_args(conf, model_and_diffusion_defaults().keys()), conf=conf
    )
    model.load_state_dict(
        dist_util.load_state_dict(os.path.expanduser(
            conf.model_path), map_location="cpu")
    )
    model.to(device)
    if conf.use_fp16:
        model.convert_to_fp16()
    model.eval()

    show_progress = conf.show_progress

    if conf.classifier_scale > 0 and conf.classifier_path:
        print("loading classifier...")
        classifier = create_classifier(
            **select_args(conf, classifier_defaults().keys()))
        classifier.load_state_dict(
            dist_util.load_state_dict(os.path.expanduser(
                conf.classifier_path), map_location="cpu")
        )

        classifier.to(device)
        if conf.classifier_use_fp16:
            classifier.convert_to_fp16()
        classifier.eval()

        def cond_fn(x, t, y=None, gt=None, **kwargs):
            assert y is not None
            with th.enable_grad():
                x_in = x.detach().requires_grad_(True)
                logits = classifier(x_in, t)
                log_probs = F.log_softmax(logits, dim=-1)
                selected = log_probs[range(len(logits)), y.view(-1)]
                return th.autograd.grad(selected.sum(), x_in)[0] * conf.classifier_scale
    else:
        cond_fn = None

    def model_fn(x, t, y=None, gt=None, **kwargs):
        assert y is not None
        return model(x, t, y if conf.class_cond else None, gt=gt)

    print("sampling...")
    all_images = []

    dset = 'eval'

    eval_name = conf.get_default_eval_name()

    dl = conf.get_dataloader(dset=dset, dsName=eval_name)

    for batch in iter(dl):

        for k in batch.keys():
            if isinstance(batch[k], th.Tensor):
                batch[k] = batch[k].to(device)

        input = batch['GT']
        print("Image size for {} is {}: ".format(batch['GT_name'], input.shape))
        mask = batch.get('gt_keep_mask')
        mask_final = batch.get('gt_keep_mask')
        mask_final = mask_final.repeat(1, 3, 1, 1)
        ### Patchifying input via Torch's unfold
        
        # kernel size for window/patch
        k = 256 
        # stride / overlap
        d = 256//2

        ### Pad images to multiple of the window size
        
        #hpadding
        hpad = (k-input.size(2)%k) // 2 
        #wpadding
        wpad = (k-input.size(3)%k) // 2 

        x = torch.nn.functional.pad(input,(wpad,wpad,hpad,hpad), mode='reflect') 
        c, h, w = x.size(1), x.size(2), x.size(3)
        mask = torch.nn.functional.pad(mask,(wpad,wpad,hpad,hpad), mode='reflect') 

        ### Unfold into patches
        patches_input = x.unfold(2, k, d).unfold(3, k, d) 
        patches_mask = mask.unfold(2, k, d).unfold(3, k, d) 
        unfold_shape = patches_input.size()
        nb_patches_h, nb_patches_w = unfold_shape[2], unfold_shape[3]

        ### Create 2D Hann windows for blending overlapping patches             
        win1d = torch.hann_window(256)
        win2d = torch.outer(win1d, win1d.t())

        window_patches = win2d.unsqueeze(0).unsqueeze(0).unsqueeze(0).unsqueeze(0).repeat(1, 3, nb_patches_h, nb_patches_w, 1, 1)
        
        window_size =256
        step = window_size >> 1
        window = win2d.numpy()
        window_u = np.vstack([np.tile(window[step:step+1, :], (step, 1)), window[step:, :]])
        window_b = np.vstack([window[:step, :], np.tile(window[step:step+1, :], (step, 1))])
        window_l = np.hstack([np.tile(window[:, step:step+1], (1, step)), window[:, step:]])
        window_r = np.hstack([window[:, :step], np.tile(window[:, step:step+1], (1, step))])
        window_ul = np.block([
            [np.ones((step, step)), window_u[:step, step:]],
            [window_l[step:, :step], window_l[step:, step:]]])
        window_ur = np.block([
            [window_u[:step, :step], np.ones((step, step))],
            [window_r[step:, :step], window_r[step:, step:]]])
        window_bl = np.block([
            [window_l[:step, :step], window_l[:step, step:]],
            [np.ones((step, step)), window_b[step:, step:]]])
        window_br = np.block([
            [window_r[:step, :step], window_r[:step, step:]],
            [window_b[step:, :step], np.ones((step, step))]])
        windows =  np.array([
            [ window_ul, window_u, window_ur ],
            [ window_l,  window,   window_r  ],
            [ window_bl, window_b, window_br ],
        ])
        
        windows = torch.from_numpy(windows)
        
        window_patches[:, :, 0, 0, :, :] = windows[0, 0]
        window_patches[:, :, -1, -1, :, :] = windows[-1, -1]
        window_patches[:, :, 0, -1, :, :] = windows[0, -1]
        window_patches[:, :, -1, 0, :, :] = windows[-1, 0]

        window_patches[:, :, 0, 1:-1, :, :] = windows[0, 1:-1]
        window_patches[:, :, -1, 1:-1, :, :] = windows[-1, 1:-1]
        window_patches[:, :, 1:-1, 0, :, :] = windows[1:-1, 0]
        window_patches[:, :, 1:-1, -1, :, :] = windows[1:-1, -1]
       
        torch.cuda.empty_cache()
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        with torch.no_grad():
            # generated = model.inference(input, mask)
            # ### Process patches individually
            # patches_input = patches_input.squeeze()
            # patches_mask = patches_mask.squeeze(0) # C x I x J x H x W
            
            # B X C x I x J x H x W
            patches_input = patches_input.permute(0, 2, 3, 1, 4, 5)
            patches_mask = patches_mask.permute(0, 2, 3, 1, 4, 5) # I x J x C x H x W
            patches_mask_copy = torch.clone(patches_mask)
            patches_mask_copy = patches_mask_copy.squeeze().contiguous().view(-1, 256, 256)
            counts_patches = [torch.numel(torch.unique(t)) for t in patches_mask_copy]
            print("There are {} patches with damage and {} patches without damage in {}.".format(counts_patches.count(2), counts_patches.count(1), batch['GT_name']))
            ### Create storage tensor for output restorations
            temp_input = torch.empty(patches_input.shape) 
            temp_sample = torch.empty(patches_input.shape) 

            for i in range(nb_patches_h):
                for j in range (0, nb_patches_w):
                    print("Processing patch [{}][{}] out of [{}][{}] for image {}".format(i, j, nb_patches_h, nb_patches_w, batch['GT_name']))
                    # temp = model.inference(
                    #     patches_input[i,j:j+8,:,:,:].to(device, dtype = torch.float),
                    #     patches_mask[i,j:j+8,:,:,:].to(device, dtype = torch.float)
                    #     )
                    model_kwargs = {}

                    model_kwargs["gt"] = patches_input[:,i,j,:,:,:].to(device, dtype = torch.float)

                    gt_keep_mask = patches_mask[:,i,j,:,:,:].to(device, dtype = torch.float)

                    if torch.numel(torch.unique(gt_keep_mask)) == 1:
                        print("There is no damage in this patch, skipping to next one...")
                        temp_input[:,i,j,:,:,:] = model_kwargs["gt"]                
                        temp_sample[:,i,j,:,:,:] = model_kwargs["gt"]
                        continue

                    if gt_keep_mask is not None:
                        model_kwargs['gt_keep_mask'] = gt_keep_mask

                    batch_size = model_kwargs["gt"].shape[0]

                    if conf.cond_y is not None:
                        classes = th.ones(batch_size, dtype=th.long, device=device)
                        model_kwargs["y"] = classes * conf.cond_y
                    else:
                        classes = th.randint(
                            low=0, high=NUM_CLASSES, size=(batch_size,), device=device
                        )
                        model_kwargs["y"] = classes

                    sample_fn = (
                        diffusion.p_sample_loop if not conf.use_ddim else diffusion.ddim_sample_loop
                    )

                    temp = sample_fn(
                        model_fn,
                        (batch_size, 3, conf.image_size, conf.image_size),
                        clip_denoised=conf.clip_denoised,
                        model_kwargs=model_kwargs,
                        cond_fn=cond_fn,
                        device=device,
                        progress=show_progress,
                        return_all=True,
                        conf=conf
                    )
                    temp_input[:,i,j,:,:,:] = temp['gt']                
                    temp_sample[:,i,j,:,:,:] = temp['sample']
        # B x I x J x C x H x W - > B x C x I x J x H x W
        temp_input = temp_input.permute(0, 3, 1, 2, 4, 5)        
        temp_sample = temp_sample.permute(0, 3, 1, 2, 4, 5)        

        temp_input = temp_input * window_patches
        temp_sample = temp_sample * window_patches
        
        temp_input = temp_input.contiguous().view(1, c, -1, k*k)
        temp_input = temp_input.permute(0, 1, 3, 2)
        temp_input = temp_input.contiguous().view(1, c*k*k, -1)

        temp_sample = temp_sample.contiguous().view(1, c, -1, k*k)
        temp_sample = temp_sample.permute(0, 1, 3, 2)
        temp_sample = temp_sample.contiguous().view(1, c*k*k, -1)

        temp_input = torch.nn.functional.fold(temp_input, output_size=(h, w), kernel_size=k, stride=d)
        temp_input = temp_input[:, :, hpad:input.size(2)+hpad, wpad:input.size(3)+wpad]

        temp_sample = torch.nn.functional.fold(temp_sample, output_size=(h, w), kernel_size=k, stride=d)
        temp_sample = temp_sample[:, :, hpad:input.size(2)+hpad, wpad:input.size(3)+wpad]

        srs = toU8((temp_sample))
        gts = toU8((temp_input))
        lrs = toU8((temp_input))


        #gt_keep_masks = toU8((model_kwargs.get('gt_keep_mask') * 2 - 1))
        gt_keep_masks = toU8(mask_final * 2 - 1)
        conf.eval_imswrite(
            srs=srs, gts=gts, lrs=lrs, gt_keep_masks=gt_keep_masks,
            img_names=batch['GT_name'], dset=dset, name=eval_name, verify_same=False)

    print("sampling complete")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--conf_path', type=str, required=False, default=None)
    args = vars(parser.parse_args())

    conf_arg = conf_mgt.conf_base.Default_Conf()
    conf_arg.update(yamlread(args.get('conf_path')))
    main(conf_arg)
