import os
import imageio
from PIL import Image

import torch
import torch.nn.functional as F

from diffusers import IFSuperResolutionPipeline, VideoToVideoSDPipeline
from diffusers.utils.torch_utils import randn_tensor

from showone.pipelines import TextToVideoIFPipeline, TextToVideoIFInterpPipeline, TextToVideoIFSuperResolutionPipeline
from showone.pipelines.pipeline_t2v_base_pixel import tensor2vid
from showone.pipelines.pipeline_t2v_sr_pixel_cond import TextToVideoIFSuperResolutionPipeline_Cond


# Base Model
# When using "showlab/show-1-base-0.0", it's advisable to increase the number of inference steps (e.g., 100) 
# and opt for a larger guidance scale (e.g., 12.0) to enhance visual quality.
pretrained_model_path = "showlab/show-1-base"
pipe_base = TextToVideoIFPipeline.from_pretrained(
    pretrained_model_path,
    torch_dtype=torch.float16,
    variant="fp16"
)
pipe_base.enable_model_cpu_offload()

# Interpolation Model
pretrained_model_path = "showlab/show-1-interpolation"
pipe_interp_1 = TextToVideoIFInterpPipeline.from_pretrained(
    pretrained_model_path, 
    torch_dtype=torch.float16, 
    variant="fp16"
)
pipe_interp_1.enable_model_cpu_offload()

# Super-Resolution Model 1
# Image super-resolution model from DeepFloyd https://huggingface.co/DeepFloyd/IF-II-L-v1.0
pretrained_model_path = "DeepFloyd/IF-II-L-v1.0"
pipe_sr_1_image = IFSuperResolutionPipeline.from_pretrained(
    pretrained_model_path,
    text_encoder=None,
    torch_dtype=torch.float16,
    variant="fp16"
)
pipe_sr_1_image.enable_model_cpu_offload()

pretrained_model_path = "showlab/show-1-sr1"
pipe_sr_1_cond = TextToVideoIFSuperResolutionPipeline_Cond.from_pretrained(
    pretrained_model_path, 
    torch_dtype=torch.float16
)
pipe_sr_1_cond.enable_model_cpu_offload()

# Super-Resolution Model 2
pretrained_model_path = "showlab/show-1-sr2"
pipe_sr_2 = VideoToVideoSDPipeline.from_pretrained(
    pretrained_model_path,
    torch_dtype=torch.float16
)
pipe_sr_2.enable_model_cpu_offload()
pipe_sr_2.enable_vae_slicing()


# Inference
prompt = "A burning lamborghini driving on rainbow."
output_dir = "./outputs/example"
negative_prompt = "low resolution, blur"

seed = 345
os.makedirs(output_dir, exist_ok=True)

# Text embeds
prompt_embeds, negative_embeds = pipe_base.encode_prompt(prompt)

# Keyframes generation (8x64x40, 2fps)
video_frames = pipe_base(
    prompt_embeds=prompt_embeds,
    negative_prompt_embeds=negative_embeds,
    num_frames=8,
    height=40,
    width=64,
    num_inference_steps=75,
    guidance_scale=9.0,
    generator=torch.manual_seed(seed),
    output_type="pt"
).frames

imageio.mimsave(f"{output_dir}/{prompt}_base.gif", tensor2vid(video_frames.clone()), fps=2)

# Frame interpolation (8x64x40, 2fps -> 29x64x40, 7.5fps)
bsz, channel, num_frames, height, width = video_frames.shape
new_num_frames = 3 * (num_frames - 1) + num_frames
new_video_frames = torch.zeros((bsz, channel, new_num_frames, height, width), 
                               dtype=video_frames.dtype, device=video_frames.device)
new_video_frames[:, :, torch.arange(0, new_num_frames, 4), ...] = video_frames
init_noise = randn_tensor((bsz, channel, 5, height, width), dtype=video_frames.dtype, 
                          device=video_frames.device, generator=torch.manual_seed(seed))

for i in range(num_frames - 1):
    batch_i = torch.zeros((bsz, channel, 5, height, width), dtype=video_frames.dtype, device=video_frames.device)
    batch_i[:, :, 0, ...] = video_frames[:, :, i, ...]
    batch_i[:, :, -1, ...] = video_frames[:, :, i + 1, ...]
    batch_i = pipe_interp_1(
        pixel_values=batch_i,
        prompt_embeds=prompt_embeds,
        negative_prompt_embeds=negative_embeds,
        num_frames=batch_i.shape[2],
        height=40,
        width=64,
        num_inference_steps=75,
        guidance_scale=4.0,
        generator=torch.manual_seed(seed),
        output_type="pt",
        init_noise=init_noise,
        cond_interpolation=True,
    ).frames

    new_video_frames[:, :, i * 4:i * 4 + 5, ...] = batch_i

video_frames = new_video_frames
imageio.mimsave(f"{output_dir}/{prompt}_interp.gif", tensor2vid(video_frames.clone()), fps=8)

# Super-resolution 1 (29x64x40 -> 29x256x160)
bsz, channel, num_frames, height, width = video_frames.shape
window_size, stride = 8, 7
new_video_frames = torch.zeros(
    (bsz, channel, num_frames, height * 4, width * 4),
    dtype=video_frames.dtype,
    device=video_frames.device)
for i in range(0, num_frames - window_size + 1, stride):
    batch_i = video_frames[:, :, i:i + window_size, ...]
    all_frame_cond = None

    if i == 0:
        first_frame_cond = pipe_sr_1_image(
            image=video_frames[:, :, 0, ...],
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_embeds,
            height=height * 4,
            width=width * 4,
            num_inference_steps=70,
            guidance_scale=4.0,
            noise_level=150,
            generator=torch.manual_seed(seed),
            output_type="pt"
        ).images
        first_frame_cond = first_frame_cond.unsqueeze(2)
    else:
        first_frame_cond = new_video_frames[:, :, i:i + 1, ...]

    batch_i = pipe_sr_1_cond(
        image=batch_i,
        prompt_embeds=prompt_embeds,
        negative_prompt_embeds=negative_embeds,
        first_frame_cond=first_frame_cond,
        height=height * 4,
        width=width * 4,
        num_inference_steps=125,
        guidance_scale=7.0,
        noise_level=250,
        generator=torch.manual_seed(seed),
        output_type="pt"
    ).frames
    new_video_frames[:, :, i:i + window_size, ...] = batch_i

video_frames = new_video_frames
imageio.mimsave(f"{output_dir}/{prompt}_sr1.gif", tensor2vid(video_frames.clone()), fps=8)

# Super-resolution 2 (29x256x160 -> 29x576x320)
video_frames = [Image.fromarray(frame).resize((576, 320)) for frame in tensor2vid(video_frames.clone())]
video_frames = pipe_sr_2(
    prompt,
    negative_prompt=negative_prompt,
    video=video_frames,
    strength=0.8,
    num_inference_steps=50,
    generator=torch.manual_seed(seed),
    output_type="pt"
).frames

imageio.mimsave(f"{output_dir}/{prompt}.gif", tensor2vid(video_frames.clone()), fps=8)
