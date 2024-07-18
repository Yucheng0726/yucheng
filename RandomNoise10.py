import argparse, os, sys, glob
import yaml 
import torch
import numpy as np
from omegaconf import OmegaConf
from PIL import Image
from tqdm import tqdm

from pytorch_lightning import seed_everything
from diffusers import DDIMScheduler, UNet2DConditionModel, VQModel, DPMSolverMultistepScheduler, DDPMScheduler
from diffusers.pipelines.latent_diffusion.pipeline_latent_diffusion import LDMClassToImagePipeline
#from eval_scripts.push_ldm_model_pretrain import convert_from_ldm_to_diffuser

def save_image(img, save_path):
    img.save(save_path, "JPEG", quality=100, subsampling=0)


def save_images_multiprocess(mp_save_count, gen_images_list, save_names_list):
    import multiprocessing
    pool = multiprocessing.Pool(mp_save_count)
    pool.starmap(save_image, zip(gen_images_list, save_names_list))
    pool.close()
    pool.join()



def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--outdir",
        type=str,
        nargs="?",
        help="dir to write results to",
        default="tmp"
    )
    parser.add_argument(
        "--ddim_steps",
        type=int,
        default=50,
        help="number of ddim sampling steps",
    )
    # parser.add_argument(
    #     "--plms",
    #     action='store_true',
    #     help="use plms sampling",
    # )
    # parser.add_argument(
    #     "--dpm_solver",
    #     action='store_true',
    #     help="use dpm_solver sampling",
    # )cd
    parser.add_argument(
        "--scheduler",
        type=str,
        default="ddim",
        choices=["ddim", "dpm"],
    )
    parser.add_argument(
        "--fixed_code",
        action='store_true',
        help="if enabled, uses the same starting code across samples ",
    )
    parser.add_argument(
        "--ddim_eta",
        type=float,
        default=0.0,
        help="ddim eta (eta=0.0 corresponds to deterministic sampling",
    )

    
    parser.add_argument(
        "--f",
        type=int,
        default=4,
        help="downsampling factor",
    )
    
    parser.add_argument(
        "--image_size",
        type=int,
        default=256,
        help='image size to genearte'
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=5,
        help="how many samples to produce for each given prompt. A.k.a. batch size",
    )
    parser.add_argument(
        "--n_samples",
        type=int,
        default=5,
        help="how many samples to produce for each given prompt. A.k.a. batch size",
    )
    parser.add_argument(
        "--scale",
        type=float,
        default=7.5,
        help="unconditional guidance scale: eps = eps(x, empty) + scale * (eps(x, cond) - eps(x, empty))",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/latent-diffusion-imagenet/train_ldm_imagenet_random_noise_50.yaml",
        help="path to config which constructs model",
    )
    parser.add_argument(
        "--ckpt",
        type=str,
        default="checkpoints/ldm/ldm_imagenet_random_noise_50",
        help="path to checkpoint of model",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="the seed (for reproducible sampling)",
    )
    parser.add_argument(
        "--precision",
        type=str,
        help="evaluate at this precision",
        choices=["full", "autocast"],
        default="autocast"
    )
    
    # split classes
    parser.add_argument(
        "--class_start_idx", type=int, help="class number start_idx", default=0
    )
    parser.add_argument(
        "--class_end_idx", type=int, help="class number end_idx", default=100
    )
    
    # push to hub
    parser.add_argument(
        '--push_to_hub',
        action='store_true',
        help='push model to hub'
    )

    parser.add_argument("--pretrained_model_name_or_path", type=str, default="Hhhhhao97/ldm_imagenet_random_noise_10")
    parser.add_argument("--hub_token", type=str, default="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")

    parser.add_argument(
        "--uncond",
        action='store_true',
        help="if set, generate images without conditioning on class labels"
    )
    
    opt = parser.parse_args()

    seed_everything(opt.seed)

    uncond = opt.uncond
    
      # load pipeline
    pipeline = LDMClassToImagePipeline.from_pretrained(opt.pretrained_model_name_or_path, use_auth_token=True, token=opt.hub_token)    
    # replace a faster scheduler
    pipeline.scheduler = DPMSolverMultistepScheduler.from_config(pipeline.scheduler.config)
    pipeline = pipeline.to('cuda')
    pipeline = pipeline.to(torch.float16)
    vqvae = pipeline.vqvae
    
    # output dir
    os.makedirs(opt.outdir, exist_ok=True)
    outpath = opt.outdir

    # get class labels
    classes = np.arange(opt.class_start_idx, opt.class_end_idx)
    n_samples_per_class = opt.n_samples
    base_count = 0
    batch_size = opt.batch_size

    with open('data/synset_human.txt', "r") as f:
        syn2name_dict = f.read().splitlines()
        syn2name_dict = dict(line.split(maxsplit=1) for line in syn2name_dict)
    with open('data/index_synset.yaml', "r") as f:
        idx2syn_dict = yaml.safe_load(f)
    
    # # push to hub
    # repo_id = '.'.join('_'.join(opt.config.split('/')[-1].split('_')[1:]).split('.')[:-1])
    # pipeline.push_to_hub(repo_id, private=True, token='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')
    
    with torch.no_grad():
        for class_label in tqdm(classes):
    
            folder_name = idx2syn_dict[class_label]
            class_human_name = syn2name_dict[folder_name].split(',')[0]
            sample_path = os.path.join(outpath, folder_name)
            os.makedirs(sample_path, exist_ok=True)
            base_count = 0
            print(f"rendering {n_samples_per_class} examples of class '{class_label}' in {opt.ddim_steps} steps and using s={opt.scale:.2f}.")
            
            for _ in range(0, n_samples_per_class, batch_size):
            
                start_code = None
                if opt.fixed_code:
                    start_code = torch.randn([batch_size, 3, 64, 64], device=torch.device('cuda')).to(torch.float16)
                
                if uncond:
                    images = pipeline(batch_size, num_inference_steps=opt.ddim_steps, eta=opt.ddim_eta, guidance_scale=opt.scale, latents=start_code).images
                else:
                    images = pipeline([class_label] * batch_size, num_inference_steps=opt.ddim_steps, eta=opt.ddim_eta, guidance_scale=opt.scale, latents=start_code).images
                
                save_names = [f"{sample_path}/{class_human_name}_{base_count+idx:05}.jpeg" for idx in range(len(images))]
                base_count += len(images)
                save_images_multiprocess(8, images, save_names)
    
    
if __name__ == '__main__':
    main()