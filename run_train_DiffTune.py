import torch
import torch.nn.functional as F
import os, argparse
from datetime import datetime
from diffusers import (
    DDPMScheduler,
    AutoencoderKL
)

from diffusers.models.unets.unet_2d_condition_diffdis import UNet2DConditionModel_diffdis
from transformers import CLIPTextModel, CLIPTokenizer
from utils.dataset_strategy import get_loader
from utils.utils import *
from utils.image_util import cutmix
from safetensors.torch import save_model
from accelerate import Accelerator
from torch.utils.tensorboard import SummaryWriter

# Khởi tạo Writer
writer = SummaryWriter('./runs/DiffDIS_FineTuneVAE')

parser = argparse.ArgumentParser()
parser.add_argument('--epoch', type=int, default=90, help='epoch number')
parser.add_argument('--lr_gen', type=float, default=1e-5, help='learning rate (lowered for finetuning)') # Giảm LR
parser.add_argument('--batchsize', type=int, default=6, help='training batch size')
parser.add_argument('--trainsize', type=int, default=1024, help='training dataset size')
parser.add_argument('--decay_rate', type=float, default=0.95, help='decay rate of learning rate')
parser.add_argument('--decay_epoch', type=int, default=30, help='every n epochs decay learning rate')
parser.add_argument("--pretrained_model_name_or_path", type=str, default='/path/to/sd-turbo/')
parser.add_argument("--dataset_path", type=str, default='/path/to/DIS5K/')
opt = parser.parse_args()
print('Generator Learning Rate: {}'.format(opt.lr_gen))

accelerator = Accelerator()

# --- 1. BUILD MODELS ---
text_encoder = CLIPTextModel.from_pretrained(opt.pretrained_model_name_or_path, subfolder='text_encoder')
vae = AutoencoderKL.from_pretrained(opt.pretrained_model_name_or_path, subfolder='vae')

# Load U-Net (Giữ nguyên để lấy khả năng dự đoán Latent)
unet = UNet2DConditionModel_diffdis.from_pretrained(opt.pretrained_model_name_or_path, subfolder="unet",
                                    in_channels=4, sample_size=96,
                                    low_cpu_mem_usage=False,
                                    ignore_mismatched_sizes=False,
                                    class_embed_type='projection',
                                    projection_class_embeddings_input_dim=4,
                                    mid_extra_cross=True,
                                    mode = 'DBIA',
                                    use_swci = True, 
                                    )

# --- 2. FREEZE / UNFREEZE LOGIC ---

# Freeze Text Encoder
text_encoder.requires_grad_(False)

# Freeze U-Net (QUAN TRỌNG: Khóa hoàn toàn U-Net)
unet.requires_grad_(False)
unet.eval() 

# Cấu hình VAE
# Freeze Encoder & Quant Conv (Không học lại cách nén)
vae.encoder.requires_grad_(False)
vae.quant_conv.requires_grad_(False)

# Unfreeze Decoder & Post Quant Conv (Học cách giải mã sắc nét)
vae.decoder.requires_grad_(True)
vae.post_quant_conv.requires_grad_(True)

vae.train() # VAE ở chế độ train

# Chuyển model sang thiết bị (Accelerator sẽ tự quản lý sau, nhưng gán trước cho chắc)
unet.to(accelerator.device)
vae.to(accelerator.device) 

# --- 3. SCHEDULER & TOKENIZER ---
noise_scheduler = DDPMScheduler.from_pretrained(opt.pretrained_model_name_or_path, subfolder='scheduler')
noise_scheduler.set_timesteps(1, device=accelerator.device)
noise_scheduler.alphas_cumprod = noise_scheduler.alphas_cumprod.to(accelerator.device)
tokenizer = CLIPTokenizer.from_pretrained(opt.pretrained_model_name_or_path,subfolder='tokenizer')

# --- 4. OPTIMIZER SETUP (CHỈ CHO DECODER) ---
# (Đã xóa đoạn code optimizer cho U-Net cũ)

decoder_params = list(vae.decoder.parameters()) + list(vae.post_quant_conv.parameters())

generator_optimizer = torch.optim.AdamW( 
    decoder_params, 
    lr=opt.lr_gen, 
    betas=(0.9, 0.999),
    weight_decay=1e-2
)

# --- 5. DATA LOADER ---
image_root = f'{opt.dataset_path}/DIS-TR/im/'
gt_root = f'{opt.dataset_path}/DIS-TR/gt/'
edge_root = f'{opt.dataset_path}/DIS-TR/contour/'

train_loader = get_loader(image_root, gt_root, edge_root, batchsize=opt.batchsize, trainsize=opt.trainsize)

# Prepare với Accelerator (Lưu ý: Prepare VAE thay vì U-Net)
vae, generator_optimizer, train_loader = accelerator.prepare(vae, generator_optimizer, train_loader)

total_step = len(train_loader)
print(total_step)

rgb_latent_scale_factor = 0.18215
weight_dtype = torch.float32
if accelerator.mixed_precision == "fp16":
    weight_dtype = torch.float16
elif accelerator.mixed_precision == "bf16":
    weight_dtype = torch.bfloat16

text_encoder.to(accelerator.device, dtype=weight_dtype)
unet.to(accelerator.device, dtype=weight_dtype) 

# THÊM DÒNG NÀY: Ép kiểu cho VAE 
# Vì VAE chỉ có Decoder được unfreeze, các phần khác như Encoder 
# cần được ép kiểu thủ công để khớp với dữ liệu đầu vào FP16
vae.to(accelerator.device, dtype=weight_dtype)

mse_loss = torch.nn.MSELoss(size_average=True, reduce=True)
size_rates = [1] 

# --- 6. TRAINING LOOP ---
for epoch in range(1, opt.epoch+1):
    loss_record = AvgMeter()
    print('Generator Learning Rate: {}'.format(generator_optimizer.param_groups[0]['lr']))
    
    for i, pack in enumerate(train_loader, start=1):
        # Accumulate gradients trên VAE
        with accelerator.accumulate(vae):
            for rate in size_rates:
                generator_optimizer.zero_grad()

                rgb, label, edge, box = pack

                rgb = rgb.to(accelerator.device).to(weight_dtype) 
                label=label.unsqueeze(1).repeat(1,3,1,1).to(accelerator.device).to(weight_dtype) 
                edge=edge.unsqueeze(1).repeat(1,3,1,1).to(accelerator.device).to(weight_dtype) 
                box=box.unsqueeze(1).repeat(1,3,1,1).to(accelerator.device).to(weight_dtype) 

                bsz = rgb.shape[0]
                assert bsz % 2 == 0, "Batch size must be even"

                rgb_chunks = rgb.chunk(bsz // 2, dim=0)       
                label_chunks = label.chunk(bsz // 2, dim=0)    
                edge_chunks = edge.chunk(bsz // 2, dim=0)      
                box_chunks = box.chunk(bsz // 2, dim=0) 

                # apply cutmix within each chunk
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
                    rgb_resized2_latents, rgb_resized4_latents, rgb_resized8_latents = generate_multi_scale_latents(rgb_mix, rgb_latent_scale_factor, vae, weight_dtype, opt)
                    
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
                    
                
                # --- GIAI ĐOẠN 2: DECODE & TÍNH LOSS PIXEL (WITH GRADIENT) ---
                
                # Scale ngược lại trước khi đưa vào Decoder (QUAN TRỌNG)
                latents_to_decode = x_denoised / rgb_latent_scale_factor
                
                # Decode ra ảnh Pixel (Decoder đang được unfreeze nên sẽ lưu gradient)
                decoded_images = vae.decode(latents_to_decode).sample
                
                # Tách ra Mask và Edge (trên không gian Pixel)
                pred_mask_pixel, pred_edge_pixel = torch.chunk(decoded_images, 2, dim=0)

                # Tính Loss trên PIXEL SPACE
                # label_mix và edge_mix là ảnh gốc (Ground Truth)
                loss1 = mse_loss(pred_mask_pixel, label_mix) 
                loss2 = mse_loss(pred_edge_pixel, edge_mix)
                
                loss = loss1 + loss2
                
                writer.add_scalar('mask_loss', loss1.item(), epoch * len(train_loader) + i)
                writer.add_scalar('edge_loss', loss2.item(), epoch * len(train_loader) + i)
                writer.add_scalar('total_loss', loss.item(), epoch * len(train_loader) + i)

                accelerator.backward(loss)
                generator_optimizer.step()
                generator_optimizer.zero_grad()
                if rate == 1:
                    loss_record.update(loss.data, opt.batchsize)

        if accelerator.is_main_process:
            if i % 10 == 0 or i == total_step:
                print('{} Epoch [{:03d}/{:03d}], Step [{:04d}/{:04d}], gen Loss: {:.4f}, mask loss:{:.4f}, edge loss:{:.4f}'.
                    format(datetime.now(), epoch, opt.epoch, i, total_step, loss_record.show(), loss1, loss2))

    if accelerator.is_main_process:

        adjust_lr(generator_optimizer, opt.lr_gen, epoch, opt.decay_rate, opt.decay_epoch)
        # --- 7. SAVE CHECKPOINTS (LƯU VAE) ---
        if epoch % 10 == 0: 
            save_path = f'../saved_model/DiffDIS_VAE_Finetune/Model_{epoch}/vae/'
            if not os.path.exists(save_path):
                os.makedirs(save_path)
            
            # Lưu VAE (chứa Decoder đã fine-tune)
            # Lưu ý: Unwrap model nếu dùng DDP
            unwrapped_vae = accelerator.unwrap_model(vae)
            unwrapped_vae.save_pretrained(save_path)
            
            # Lưu Optimizer state
            optimizer_state = generator_optimizer.state_dict()
            torch.save(optimizer_state, f'../saved_model/DiffDIS_VAE_Finetune/Model_{epoch}/generator_optimizer.pth')