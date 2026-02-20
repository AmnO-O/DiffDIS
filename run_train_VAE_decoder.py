import argparse
from csv import writer
from datetime import datetime
import datetime
import os
import logging
import glob

import numpy as np
import torch
from PIL import Image
from tqdm.auto import tqdm

from core.diffdis_pipeline import DiffDISPipeline
from diffusers import (
    DDPMScheduler,
    AutoencoderKL,
)

from diffusers.models.unets.unet_2d_condition_diffdis import UNet2DConditionModel_diffdis

from transformers import CLIPTextModel, CLIPTokenizer

from utils.dataset_strategy import get_loader
from utils.seed_all import seed_all 
from utils.utils import check_mkdir
from utils.config import diste1,diste2,diste3,diste4,disvd
from utils.image_util import resize_res
from utils.utils import *
from utils.image_util import cutmix

from accelerate import Accelerator
accelerator = Accelerator()


from torchvision import transforms
to_pil = transforms.ToPILImage()


## DIS dataset
to_test ={
    'DIS-VD':disvd,
    'DIS-TE1':diste1,
    'DIS-TE2':diste2,
    'DIS-TE3':diste3,
    'DIS-TE4':diste4,
}

Image.MAX_IMAGE_PIXELS = None # Cho phép xử lý ảnh siêu lớn

EXTENSION_LIST = [".jpg", ".jpeg", ".png"]

if __name__=="__main__":
    
    use_seperate = False

    logging.basicConfig(level=logging.INFO)
    
    '''Set the Args'''
    parser = argparse.ArgumentParser()

    parser.add_argument("--checkpoint_path", type=str, default='/path/to/your/unet/')    
    parser.add_argument("--pretrained_model_path", type=str, default='/path/to/pretrained/models/')  
    parser.add_argument("--output_dir", type=str, default='/path/to/save/outputs/')   

    '''Add more parameters for training VAE decoder'''
# --- Training Hyperparameters ---
    parser.add_argument('--epoch', type=int, default=90, help='Total training epochs')
    parser.add_argument('--lr_gen', type=float, default=1e-5, help='Learning rate for VAE decoder')
    parser.add_argument('--train_batch_size', type=int, default=6, help='Training batch size')
    parser.add_argument('--trainsize', type=int, default=1024, help='Input resolution for training')
    parser.add_argument('--decay_rate', type=float, default=0.95, help='Decay rate of learning rate')
    parser.add_argument('--decay_epoch', type=int, default=30, help='Every n epochs decay learning rate')
    
    # --- Paths ---
    parser.add_argument("--dataset_path", type=str, 
                        default='/path/to/DIS5K/', help="Path to DIS5K dataset")
    

    # inference setting
    parser.add_argument(
        "--denoise_steps",
        type=int,
        default=1,
        help="Diffusion denoising steps, more steps results in higher accuracy but slower inference speed.",
    )
    parser.add_argument(
        "--ensemble_size",
        type=int,
        default=1,
        help="Number of predictions to be ensembled, more inference gives better results but runs slower.",
    )
    parser.add_argument(
        "--half_precision",
        action="store_true",
        help="Run with half-precision (16-bit float), might lead to suboptimal result.",
    )

    # resolution setting
    parser.add_argument(
        "--processing_res",
        type=int,
        default=1024,
        help="Maximum resolution of processing. 0 for using input image resolution. Default: 768.",
    )
    parser.add_argument(
        "--output_processing_res",
        action="store_true",
        help="When input is resized, output depth at resized operating resolution. Default: False.",
    )

    # other settings
    parser.add_argument("--seed", type=int, default=None, help="Random seed.")
    parser.add_argument(
        "--test_batch_size",
        type=int,
        default=0,
        help="Inference batch size. Default: 0 (will be set automatically).",
    )
    
    parser.add_argument("--mode", type=str, default="inference", 
                        choices=["inference", "train"], 
                        help="Mode : inference or training")

    args = parser.parse_args()
    
    pretrained_model_path = args.pretrained_model_path
    checkpoint_path = args.checkpoint_path
    output_dir = args.output_dir
    denoise_steps = args.denoise_steps
    ensemble_size = args.ensemble_size
    
    if ensemble_size>15:
        logging.warning("long ensemble steps, low speed..")
    
    half_precision = args.half_precision

    processing_res = args.processing_res
    match_input_res = not args.output_processing_res

    seed = args.seed
    batch_size = args.test_batch_size
    check_mkdir(output_dir)
    
    if batch_size==0:
        batch_size = 1  # set default batchsize
    # -------------------- Preparation --------------------
    # Random seed
    if seed is None:
        import time
        seed = int(time.time())
    seed_all(seed)

    # -------------------- Device --------------------
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
        logging.warning("CUDA is not available. Running on CPU will be slow.")
    logging.info(f"device = {device}")

    import ttach as tta
    transforms = tta.Compose(
    [
        tta.HorizontalFlip(),
        tta.Scale(scales=[0.75, 1, 1.25], interpolation='bilinear', align_corners=False),
    ])
    

    # -------------------- Model --------------------
    if half_precision:
        dtype = torch.float16
        logging.info(f"Running with half precision ({dtype}).")
    else:
        dtype = torch.float32

    vae = AutoencoderKL.from_pretrained(pretrained_model_path,subfolder='vae')
    scheduler = DDPMScheduler.from_pretrained(pretrained_model_path,subfolder='scheduler')
    text_encoder = CLIPTextModel.from_pretrained(pretrained_model_path,subfolder='text_encoder')
    tokenizer = CLIPTokenizer.from_pretrained(pretrained_model_path,subfolder='tokenizer')
    unet = UNet2DConditionModel_diffdis.from_pretrained(checkpoint_path,subfolder="unet",
                                    in_channels=8, sample_size=96,
                                    low_cpu_mem_usage=False,
                                    ignore_mismatched_sizes=False,
                                    class_embed_type='projection',
                                    projection_class_embeddings_input_dim=4,
                                    mid_extra_cross=True,
                                    mode = 'DBIA',
                                    use_swci = True)
    pipe = DiffDISPipeline(unet=unet,
                            vae=vae,
                            scheduler=scheduler,
                            text_encoder=text_encoder,
                            tokenizer=tokenizer)
    print("Using Seperated Modules")
    
    logging.info("loading pipeline whole successfully.")

    pipe = pipe.to(accelerator.device)



    # Model logic
        # --- SCHEDULER & TOKENIZER ---
    mse_loss = torch.nn.MSELoss(size_average=True, reduce=True)

    noise_scheduler = DDPMScheduler.from_pretrained(args.pretrained_model_path, subfolder='scheduler')
    noise_scheduler.set_timesteps(1, device=accelerator.device)
    noise_scheduler.alphas_cumprod = noise_scheduler.alphas_cumprod.to(accelerator.device)
    tokenizer = CLIPTokenizer.from_pretrained(args.pretrained_model_path,subfolder='tokenizer')

    rgb_latent_scale_factor = 0.18215

    image_root = f'{args.dataset_path}/DIS-TR/im/'
    gt_root = f'{args.dataset_path}/DIS-TR/gt/'
    edge_root = f'{args.dataset_path}/DIS-TR/contour/'

    train_loader = get_loader(image_root, gt_root, edge_root, batchsize=args.train_batch_size, trainsize=args.trainsize)
    total_step = len(train_loader)

    vae.requires_grad_(False)
    unet.requires_grad_(False)
    text_encoder.requires_grad_(False)

    if args.mode == 'train':
        vae.decoder.requires_grad_(True)
        optimizers = torch.optim.AdamW(
            vae.decoder.parameters(), 
            lr=args.lr_gen, 
            weight_decay=1e-2,
            betas=(0.9, 0.999)
        )

        scheduler_lr = torch.optim.lr_scheduler.StepLR(
            optimizers, 
            step_size=args.decay_epoch, 
            gamma=args.decay_rate
        )
        
        unet, vae, optimizers, train_loader = accelerator.prepare(unet, vae, optimizers, train_loader)
        vae.train()
        unet.eval()
        
        for epoch in range(args.epoch):
            loss_record = AvgMeter()
            print('Generator Learning Rate: {}'.format(optimizers.param_groups[0]['lr']))

            
            for i, pack in enumerate(train_loader, start=1):
                optimizers.zero_grad()

                rgb, label, edge, box = pack
                # Đảm bảo dùng đúng tên biến dtype bạn đã định nghĩa ở trên
                weight_dtype = dtype 

                # Cách tối ưu nhất
                rgb   = rgb.to(accelerator.device).to(weight_dtype) 
                label = label.unsqueeze(1).repeat(1,3,1,1).to(accelerator.device).to(weight_dtype) 
                edge  = edge.unsqueeze(1).repeat(1,3,1,1).to(accelerator.device).to(weight_dtype) 
                box   = box.unsqueeze(1).repeat(1,3,1,1).to(accelerator.device).to(weight_dtype)


                bsz = rgb.shape[0]
                assert bsz % 2 == 0, "Batch size must be even"

                rgb_chunks = rgb.chunk(bsz // 2, dim=0)       
                label_chunks = label.chunk(bsz // 2, dim=0)    
                edge_chunks = edge.chunk(bsz // 2, dim=0)      
                box_chunks = box.chunk(bsz // 2, dim=0) 
                
                rgbs, labels, edges = [], [], []
                for rgb_, label_, edge_, box_ in zip(rgb_chunks, label_chunks, edge_chunks, box_chunks):
                    mixed_rgb, mixed_label, mixed_edge = cutmix(rgb_, label_, edge_, box_)
                    rgbs.append(mixed_rgb)
                    labels.append(mixed_label)
                    edges.append(mixed_edge)

                rgb_mix = torch.cat(rgbs, dim=0)
                label_mix = torch.cat(labels, dim=0)
                edge_mix = torch.cat(edges, dim=0)

                # --- GIAI ĐOẠN 1: DỰ ĐOÁN LATENT (NO GRADIENT) ---
                # Chúng ta dùng U-Net đã train để lấy Latent tốt nhất có thể
                with torch.no_grad():    
                    # map pixels into latent space
                    h_batch = vae.encoder(torch.cat((rgb_mix, label_mix, edge_mix), dim=0).to(weight_dtype))
                    moments_batch = vae.quant_conv(h_batch)
                    mean_batch, logvar_batch = torch.chunk(moments_batch, 2, dim=1)
                    batch_latents = mean_batch * rgb_latent_scale_factor
                    rgb_latents, mask_latents, edge_latents = torch.chunk(batch_latents, 3, dim=0)
                    
                    # generate multi-scale conditions
                    rgb_resized2_latents, rgb_resized4_latents, rgb_resized8_latents = generate_multi_scale_latents(rgb_mix, rgb_latent_scale_factor, vae, weight_dtype, args)
                    
                    # concat mask and edge latents along batch dimension 
                    unified_latents = torch.cat((mask_latents,edge_latents), dim=0)

                    # create multi-resolution noise
                    noise = pyramid_noise_like(unified_latents, discount=0.8) 

                    # set timestep to T
                    timesteps = torch.tensor([999], device=accelerator.device).long()
                    
                    # add noise 
                    noisy_unified_latents = noise_scheduler.add_noise(unified_latents, noise, timesteps.repeat(bsz*2))
        
                    # encode text embedding for empty prompt
                    prompt = ""
                    text_inputs =tokenizer(
                        prompt,
                        padding="do_not_pad",
                        max_length=tokenizer.model_max_length,
                        truncation=True,
                        return_tensors="pt",
                    )
                    text_input_ids = text_inputs.input_ids.to(text_encoder.device) 
                    empty_text_embed = text_encoder(text_input_ids)[0].to(weight_dtype)
                    batch_empty_text_embed = empty_text_embed.repeat((noisy_unified_latents.shape[0], 1, 1))  

                    # batch discriminative embedding
                    discriminative_label = torch.tensor([[0, 1], [1, 0]], dtype=weight_dtype, device=accelerator.device)
                    BDE = torch.cat([torch.sin(discriminative_label), torch.cos(discriminative_label)], dim=-1).repeat_interleave(bsz, 0)
                    unet_input = torch.cat([rgb_latents.repeat(2,1,1,1),noisy_unified_latents], dim=1)  
                    
                    # predict the noise (U-Net Forward)
                    noise_pred = unet(unet_input, timesteps.repeat(bsz*2), encoder_hidden_states=batch_empty_text_embed, class_labels = BDE,\
                                    rgb_token=[rgb_latents.repeat(2,1,1,1) , rgb_resized2_latents, rgb_resized4_latents, rgb_resized8_latents],\
                        ).sample 
                    
                    # one-step denoising process (Lấy Latent đã khử nhiễu)
                    x_denoised = noise_scheduler.step(noise_pred, timesteps, noisy_unified_latents, return_dict=True).prev_sample
                                    

                # --- GIAI ĐOẠN 2: TRAIN DECODER (GRADIENT ON) ---
                latents_to_decode = x_denoised / 0.18215 # Dùng hằng số scale của VAE
                
                # Decode ra pixel space
                decoded_images = vae.decode(latents_to_decode).sample # [Batch*2, 3, 1024, 1024]
                pred_mask_pixel, pred_edge_pixel = torch.chunk(decoded_images, 2, dim=0)

                # Chuẩn hóa Ground Truth về [-1, 1] để khớp với VAE output
                gt_mask = label_mix * 2.0 - 1.0
                gt_edge = edge_mix * 2.0 - 1.0

                # Đưa về 1 kênh để tính Loss chính xác cho Segmentation
                loss1 = mse_loss(pred_mask_pixel.mean(dim=1, keepdim=True), gt_mask[:, 0:1, :, :]) 
                loss2 = mse_loss(pred_edge_pixel.mean(dim=1, keepdim=True), gt_edge[:, 0:1, :, :])
                
                loss = loss1 + loss2
                
                # Backward và Update
                accelerator.backward(loss)
                optimizers.step()
                optimizers.zero_grad()
                
                #torch.cuda.empty_cache()

                # Cập nhật record (Dùng args.train_batch_size thay vì args.batchsize nếu bạn đặt tên vậy)
                loss_record.update(loss.item(), args.train_batch_size)
                
                if accelerator.is_main_process:
                    if i % 10 == 0 or i == total_step:
                        print('{} Epoch [{:03d}/{:03d}], Step [{:04d}/{:04d}], gen Loss: {:.4f}, mask loss:{:.4f}, edge loss:{:.4f}'.
                            format(datetime.now(), epoch, args.epoch, i, total_step, loss_record.show(), loss1.item(), loss2.item()))
    
            # Sau khi kết thúc vòng lặp "for i, pack in enumerate(train_loader):"
            scheduler_lr.step() 
        
            if accelerator.is_main_process:
                # CHIẾN THUẬT LƯU: Mỗi 10 epoch hoặc epoch cuối cùng
                if epoch % 10 == 0 or epoch == args.epoch - 1:
                    save_path = os.path.join(args.output_dir, f'checkpoint-epoch-{epoch}')
                    check_mkdir(save_path)

                    # Giải phóng mô hình khỏi phân tán (unwrap)
                    unwrapped_vae = accelerator.unwrap_model(vae)
                    
                    # --- CÁCH 1: Lưu file .pth (SIÊU NHẸ, khuyên dùng) ---
                    # File này chỉ chứa trọng số Decoder, tầm 100-200MB
                    torch.save(unwrapped_vae.decoder.state_dict(), os.path.join(save_path, "vae_decoder.pth"))
                    torch.save(unwrapped_vae.post_quant_conv.state_dict(), os.path.join(save_path, "post_quant_conv.pth"))

                    # --- CÁCH 2: Lưu toàn bộ Pipeline (RẤT NẶNG, >2GB) ---
                    # Chỉ nên lưu ở epoch cuối cùng để tiết kiệm Drive
                    if epoch == args.epoch - 1:
                        pipe.vae = unwrapped_vae
                        pipe.save_pretrained(save_path)
                        logging.info(f"Full Pipeline saved at final epoch: {save_path}")
                    
                    logging.info(f"VAE weights saved to {save_path}")

        logging.info("Training Finished!")