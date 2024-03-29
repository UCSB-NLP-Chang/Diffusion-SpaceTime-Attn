# plms_single_optimize_useBothLocalAndGlobalCLIPLoss.py
"""SAMPLING ONLY."""

import torch
import numpy as np
from tqdm import tqdm
from functools import partial

from ldm.modules.diffusionmodules.util import make_ddim_sampling_parameters, make_ddim_timesteps, noise_like

import clip
import os
from PIL import Image
from einops import rearrange

import pickle as pkl
import torchvision.transforms as transforms

mode = "fix_radius_0p2"

class DCLIPLoss(torch.nn.Module):
    def __init__(self):
        super(DCLIPLoss, self).__init__()
        self.model, self.preprocess = clip.load("ViT-B/32", device="cuda")
        self.upsample = torch.nn.Upsample(scale_factor=7)
        self.avg_pool = torch.nn.AvgPool2d(kernel_size=16)
        self.transforms = torch.nn.Sequential(transforms.Resize((224,224)))

    def forward_3(self, image, text):
        image1 = image.unsqueeze(0).cuda()
        image1 = self.transforms(image).unsqueeze(0)
        text1 = clip.tokenize([text]).to("cuda")
        image1_feat = self.model.encode_image(image1)
        text1_feat = self.model.encode_text(text1)
        similarity = torch.nn.CosineSimilarity()(image1_feat, text1_feat)
        return 1 - similarity

    def forward_2(self, image, text):
        text1 = clip.tokenize([text]).to("cuda")
        image1 = image.unsqueeze(0).cuda()
        image1 = self.avg_pool(self.upsample(image1))
        image1_feat = self.model.encode_image(image1)
        text1_feat = self.model.encode_text(text1)
        similarity = torch.nn.CosineSimilarity()(image1_feat, text1_feat)
        return 1 - similarity

    def forward(self, image1, image2, text1, text2):
        text1 = clip.tokenize([text1]).to("cuda")
        text2 = clip.tokenize([text2]).to("cuda")
        image1 = image1.unsqueeze(0).cuda()
        image2 = image2.unsqueeze(0)
        image1 = self.avg_pool(self.upsample(image1))
        image2 = self.avg_pool(self.upsample(image2))
        image1_feat = self.model.encode_image(image1)
        image2_feat = self.model.encode_image(image2)
        text1_feat = self.model.encode_text(text1)
        text2_feat = self.model.encode_text(text2)
        d_image_feat = image1_feat - image2_feat
        d_text_feat = text1_feat - text2_feat
        similarity = torch.nn.CosineSimilarity()(d_image_feat, d_text_feat)
        return 1 - similarity



class PLMSSampler(object):
    def __init__(self, model, schedule="linear", **kwargs):
        super().__init__()
        self.model = model
        self.ddpm_num_timesteps = model.num_timesteps
        self.schedule = schedule
        self.clip_loss_model = DCLIPLoss()
        self.clip_loss_model.requires_grad_(False)
        # self.dino_loss = DINOLoss()

    def register_buffer(self, name, attr):
        if type(attr) == torch.Tensor:
            if attr.device != torch.device("cuda"):
                attr = attr.to(torch.device("cuda"))
        setattr(self, name, attr)

    def make_schedule(self, ddim_num_steps, ddim_discretize="uniform", ddim_eta=0., verbose=True):
        if ddim_eta != 0:
            raise ValueError('ddim_eta must be 0 for PLMS')
        self.ddim_timesteps = make_ddim_timesteps(ddim_discr_method=ddim_discretize, num_ddim_timesteps=ddim_num_steps,
                                                  num_ddpm_timesteps=self.ddpm_num_timesteps,verbose=verbose)
        alphas_cumprod = self.model.alphas_cumprod
        assert alphas_cumprod.shape[0] == self.ddpm_num_timesteps, 'alphas have to be defined for each timestep'
        to_torch = lambda x: x.clone().detach().to(torch.float32).to(self.model.device)

        self.register_buffer('betas', to_torch(self.model.betas))
        self.register_buffer('alphas_cumprod', to_torch(alphas_cumprod))
        self.register_buffer('alphas_cumprod_prev', to_torch(self.model.alphas_cumprod_prev))

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.register_buffer('sqrt_alphas_cumprod', to_torch(np.sqrt(alphas_cumprod.cpu())))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', to_torch(np.sqrt(1. - alphas_cumprod.cpu())))
        self.register_buffer('log_one_minus_alphas_cumprod', to_torch(np.log(1. - alphas_cumprod.cpu())))
        self.register_buffer('sqrt_recip_alphas_cumprod', to_torch(np.sqrt(1. / alphas_cumprod.cpu())))
        self.register_buffer('sqrt_recipm1_alphas_cumprod', to_torch(np.sqrt(1. / alphas_cumprod.cpu() - 1)))

        # ddim sampling parameters
        ddim_sigmas, ddim_alphas, ddim_alphas_prev = make_ddim_sampling_parameters(alphacums=alphas_cumprod.cpu(),
                                                                                   ddim_timesteps=self.ddim_timesteps,
                                                                                   eta=ddim_eta,verbose=verbose)
        self.register_buffer('ddim_sigmas', ddim_sigmas)
        self.register_buffer('ddim_alphas', ddim_alphas)
        self.register_buffer('ddim_alphas_prev', ddim_alphas_prev)
        self.register_buffer('ddim_sqrt_one_minus_alphas', np.sqrt(1. - ddim_alphas))
        sigmas_for_original_sampling_steps = ddim_eta * torch.sqrt(
            (1 - self.alphas_cumprod_prev) / (1 - self.alphas_cumprod) * (
                        1 - self.alphas_cumprod / self.alphas_cumprod_prev))
        self.register_buffer('ddim_sigmas_for_original_num_steps', sigmas_for_original_sampling_steps)

    def sample(self,
               S,
               batch_size,
               shape,
               conditioning=None,
               callback=None,
               normals_sequence=None,
               img_callback=None,
               quantize_x0=False,
               eta=0.,
               mask=None,
               x0=None,
               temperature=1.,
               noise_dropout=0.,
               score_corrector=None,
               corrector_kwargs=None,
               verbose=True,
               x_T=None,
               log_every_t=100,
               unconditional_guidance_scale=1.,
               unconditional_conditioning=None,
               text_index=None,
               curr_text="",
               bboxs_curr=None,
               seed=None,
               prompt_idx=None,
               object_names=None,
               # this has to come in the same format as the conditioning, # e.g. as encoded tokens, ...
               **kwargs
               ):
        if conditioning is not None:
            if isinstance(conditioning, dict):
                cbs = conditioning[list(conditioning.keys())[0]].shape[0]
                if cbs != batch_size:
                    print(f"Warning: Got {cbs} conditionings but batch-size is {batch_size}")
            else:
                if conditioning.shape[0] != batch_size:
                    print(f"Warning: Got {conditioning.shape[0]} conditionings but batch-size is {batch_size}")

        self.make_schedule(ddim_num_steps=S, ddim_eta=eta, verbose=verbose)
        # sampling
        C, H, W = shape
        size = (batch_size, C, H, W)
        print(f'Data shape for PLMS sampling is {size}')

        self.plms_sampling(conditioning, size,
            callback=callback,
            img_callback=img_callback,
            quantize_denoised=quantize_x0,
            mask=mask, x0=x0,
            ddim_use_original_steps=False,
            noise_dropout=noise_dropout,
            temperature=temperature,
            score_corrector=score_corrector,
            corrector_kwargs=corrector_kwargs,
            x_T=x_T,
            log_every_t=log_every_t,
            unconditional_guidance_scale=unconditional_guidance_scale,
            unconditional_conditioning=unconditional_conditioning,
            text_index=text_index,
            curr_text=curr_text,
            bboxs_curr=bboxs_curr,
            seed=seed,
            prompt_idx=prompt_idx,
            object_names=object_names
            )
        return None

    def plms_sampling(self, cond, shape,
                      x_T=None, ddim_use_original_steps=False,
                      callback=None, timesteps=None, quantize_denoised=False,
                      mask=None, x0=None, img_callback=None, log_every_t=100,
                      temperature=1., noise_dropout=0., score_corrector=None, corrector_kwargs=None,
                      unconditional_guidance_scale=1., unconditional_conditioning=None,text_index=None,curr_text="",bboxs_curr=None,
                      seed=None, prompt_idx=None, object_names=None):
        assert seed is not None
        assert len(bboxs_curr) == len(object_names)
        device = self.model.betas.device
        b = shape[0]
        if x_T is None:
            img = torch.randn(shape, device=device)
        else:
            img = x_T
        img_input = img.clone()
        if timesteps is None:
            timesteps = self.ddpm_num_timesteps if ddim_use_original_steps else self.ddim_timesteps
        elif timesteps is not None and not ddim_use_original_steps:
            subset_end = int(min(timesteps / self.ddim_timesteps.shape[0], 1) * self.ddim_timesteps.shape[0]) - 1
            timesteps = self.ddim_timesteps[:subset_end]

        weight_initialize_coef = 5.0
        if len(bboxs_curr) != 0:
            weighting_parameter = torch.tensor([1 / len(bboxs_curr) * weight_initialize_coef] * 50 * len(bboxs_curr)).to(img).float()
        else:
            weighting_parameter = torch.tensor([]).to(img).float()
        weighting_parameter = weighting_parameter.reshape(-1, 50)
        weighting_parameter.requires_grad = True
        from torch import optim

        initial_lr = 0.005
        optimizer = optim.Adam([weighting_parameter], lr=0.005)

        weighting_parameter_pass = weighting_parameter

        print("Optimizing start")

        for epoch in tqdm(range(3)):
            total_steps = timesteps if ddim_use_original_steps else timesteps.shape[0]
            time_range = list(reversed(range(0,timesteps))) if ddim_use_original_steps else np.flip(timesteps)
            iterator = time_range
            old_eps = []
            img = img_input.clone()
            
            for i, step in enumerate(iterator):
                index = total_steps - i - 1
                ts = torch.full((b,), step, device=device, dtype=torch.long)
                ts_next = torch.full((b,), time_range[min(i + 1, len(time_range) - 1)], device=device, dtype=torch.long)

                if mask is not None:
                    assert x0 is not None
                    img_orig = self.model.q_sample(x0, ts)  # TODO: deterministic forward pass?
                    img = img_orig * mask + (1. - mask) * img

                outs = self.p_sample_plms(img, cond, ts, index=index, use_original_steps=ddim_use_original_steps,
                                        quantize_denoised=quantize_denoised, temperature=temperature,
                                        noise_dropout=noise_dropout, score_corrector=score_corrector,
                                        corrector_kwargs=corrector_kwargs,
                                        unconditional_guidance_scale=unconditional_guidance_scale,
                                        unconditional_conditioning=unconditional_conditioning,
                                        old_eps=old_eps, t_next=ts_next, text_index=text_index, coef=weighting_parameter_pass[:,i], bboxs_curr=bboxs_curr)
                img, pred_x0, e_t = outs
                old_eps.append(e_t)
                if len(old_eps) >= 4:
                    old_eps.pop(0)

            img_temp = self.model.decode_first_stage(img)
            img_temp_ddim = torch.clamp((img_temp + 1.0) / 2.0, min=0.0, max=1.0)
            
            loss = self.clip_loss_model.forward_2(
                img_temp_ddim[0].float().cuda(), curr_text
            )
            
            for each, namei in zip(bboxs_curr, object_names):
                x_center = each[0]
                y_center = each[1]
                x1 = x_center - 0.2
                x2 = x_center + 0.2
                y1 = y_center - 0.2
                y2 = y_center + 0.2
                x1 = max(x1, 0)
                x2 = min(x2, 1)
                y1 = max(y1, 0)
                y2 = min(y2, 1)
                object_name = namei.lower()
                object_name = object_name.replace("the ","")
                loss_curr = self.clip_loss_model.forward_3(
                    img_temp_ddim[0].float().cuda()[:, int(512*y1):int(512*y2), int(512*x1):int(512*x2)], "A photo of " + object_name
                )
                print(object_name, loss_curr)
                loss+=5*loss_curr

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # save image
            if epoch == 2:
                with torch.no_grad():
                    x_sample = 255.0 * rearrange(
                        img_temp_ddim[0].detach().cpu().numpy(), "c h w -> h w c"
                    )
                    imgsave = Image.fromarray(x_sample.astype(np.uint8))
                    save_path = "result_outputs/"
                    os.makedirs(save_path, exist_ok=True)
                    imgsave.save(save_path + "final%d_s%d_index_%d.png"%(epoch, seed, prompt_idx))
                    pass

            torch.cuda.empty_cache()

        return None

    # @torch.no_grad()
    def p_sample_plms(self, x, c, t, index, repeat_noise=False, use_original_steps=False, quantize_denoised=False,
                      temperature=1., noise_dropout=0., score_corrector=None, corrector_kwargs=None,
                      unconditional_guidance_scale=1., unconditional_conditioning=None, old_eps=None, t_next=None, text_index=None, coef=None, bboxs_curr=None):
        b, *_, device = *x.shape, x.device
        def get_model_output(x, t, text_index, bboxs_curr):
            if unconditional_conditioning is None or unconditional_guidance_scale == 1.:
                e_t = self.model.apply_model(x, t, c)
            else:
                x_in = torch.cat([x] * 2)
                t_in = torch.cat([t] * 2)
                c_in = torch.cat([unconditional_conditioning, c])
                e_t_uncond, e_t = self.model.apply_model_extra(x_in, text_index, t_in, c_in, coef=coef, bboxs_curr=bboxs_curr).chunk(2)
                e_t = e_t_uncond + unconditional_guidance_scale * (e_t - e_t_uncond)

            if score_corrector is not None:
                assert self.model.parameterization == "eps"
                e_t = score_corrector.modify_score(self.model, e_t, x, t, c, **corrector_kwargs)

            return e_t

        alphas = self.model.alphas_cumprod if use_original_steps else self.ddim_alphas
        alphas_prev = self.model.alphas_cumprod_prev if use_original_steps else self.ddim_alphas_prev
        sqrt_one_minus_alphas = self.model.sqrt_one_minus_alphas_cumprod if use_original_steps else self.ddim_sqrt_one_minus_alphas
        sigmas = self.model.ddim_sigmas_for_original_num_steps if use_original_steps else self.ddim_sigmas

        def get_x_prev_and_pred_x0(e_t, index):
            # select parameters corresponding to the currently considered timestep
            a_t = torch.full((b, 1, 1, 1), alphas[index], device=device)
            a_prev = torch.full((b, 1, 1, 1), alphas_prev[index], device=device)
            sigma_t = torch.full((b, 1, 1, 1), sigmas[index], device=device)
            sqrt_one_minus_at = torch.full((b, 1, 1, 1), sqrt_one_minus_alphas[index],device=device)

            # current prediction for x_0
            pred_x0 = (x - sqrt_one_minus_at * e_t) / a_t.sqrt()
            if quantize_denoised:
                pred_x0, _, *_ = self.model.first_stage_model.quantize(pred_x0)
            # direction pointing to x_t
            dir_xt = (1. - a_prev - sigma_t**2).sqrt() * e_t
            noise = sigma_t * noise_like(x.shape, device, repeat_noise) * temperature
            if noise_dropout > 0.:
                noise = torch.nn.functional.dropout(noise, p=noise_dropout)
            x_prev = a_prev.sqrt() * pred_x0 + dir_xt + noise
            return x_prev, pred_x0

        e_t = get_model_output(x, t, text_index, bboxs_curr)
        if len(old_eps) == 0:
            # Pseudo Improved Euler (2nd order)
            x_prev, pred_x0 = get_x_prev_and_pred_x0(e_t, index)
            e_t_next = get_model_output(x_prev, t_next, text_index, bboxs_curr)
            e_t_prime = (e_t + e_t_next) / 2
        elif len(old_eps) == 1:
            # 2nd order Pseudo Linear Multistep (Adams-Bashforth)
            e_t_prime = (3 * e_t - old_eps[-1]) / 2
        elif len(old_eps) == 2:
            # 3nd order Pseudo Linear Multistep (Adams-Bashforth)
            e_t_prime = (23 * e_t - 16 * old_eps[-1] + 5 * old_eps[-2]) / 12
        elif len(old_eps) >= 3:
            # 4nd order Pseudo Linear Multistep (Adams-Bashforth)
            e_t_prime = (55 * e_t - 59 * old_eps[-1] + 37 * old_eps[-2] - 9 * old_eps[-3]) / 24

        x_prev, pred_x0 = get_x_prev_and_pred_x0(e_t_prime, index)

        return x_prev, pred_x0, e_t
