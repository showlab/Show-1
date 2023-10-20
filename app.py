import gradio as gr
import torch
from diffusers.utils import export_to_video

import os
from PIL import Image

import torch.nn.functional as F

from diffusers import IFSuperResolutionPipeline, VideoToVideoSDPipeline
from diffusers.utils import export_to_video
from diffusers.utils.torch_utils import randn_tensor

from showone.pipelines import TextToVideoIFPipeline, TextToVideoIFInterpPipeline, TextToVideoIFSuperResolutionPipeline
from showone.pipelines.pipeline_t2v_base_pixel import tensor2vid
from showone.pipelines.pipeline_t2v_sr_pixel_cond import TextToVideoIFSuperResolutionPipeline_Cond


# Base Model
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
    text_encoder=None,
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
    variant="fp16",
)
pipe_sr_1_image.enable_model_cpu_offload()

pretrained_model_path = "showlab/show-1-sr1"
pipe_sr_1_cond = TextToVideoIFSuperResolutionPipeline_Cond.from_pretrained(
    pretrained_model_path, 
    text_encoder=None,
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

output_dir = "./outputs"
os.makedirs(output_dir, exist_ok=True)

def infer(prompt):
    print(prompt)
    negative_prompt = "low resolution, blur"
    
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
        output_type="pt"
    ).frames

    # Frame interpolation (8x64x40, 2fps -> 29x64x40, 7.5fps)
    bsz, channel, num_frames, height, width = video_frames.shape
    new_num_frames = 3 * (num_frames - 1) + num_frames
    new_video_frames = torch.zeros((bsz, channel, new_num_frames, height, width), 
                                dtype=video_frames.dtype, device=video_frames.device)
    new_video_frames[:, :, torch.arange(0, new_num_frames, 4), ...] = video_frames
    init_noise = randn_tensor((bsz, channel, 5, height, width), dtype=video_frames.dtype, 
                            device=video_frames.device)

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
            num_inference_steps=50,
            guidance_scale=4.0,
            output_type="pt",
            init_noise=init_noise,
            cond_interpolation=True,
        ).frames

        new_video_frames[:, :, i * 4:i * 4 + 5, ...] = batch_i

    video_frames = new_video_frames

    # Super-resolution 1 (29x64x40 -> 29x256x160)
    bsz, channel, num_frames, height, width = video_frames.shape
    window_size, stride = 8, 7
    new_video_frames = torch.zeros(
        (bsz, channel, num_frames, height * 4, width * 4),
        dtype=video_frames.dtype,
        device=video_frames.device)
    for i in range(0, num_frames - window_size + 1, stride):
        batch_i = video_frames[:, :, i:i + window_size, ...]

        if i == 0:
            first_frame_cond = pipe_sr_1_image(
                image=video_frames[:, :, 0, ...],
                prompt_embeds=prompt_embeds,
                negative_prompt_embeds=negative_embeds,
                height=height * 4,
                width=width * 4,
                num_inference_steps=50,
                guidance_scale=4.0,
                noise_level=150,
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
            num_inference_steps=50,
            guidance_scale=7.0,
            noise_level=250,
            output_type="pt"
        ).frames
        new_video_frames[:, :, i:i + window_size, ...] = batch_i

    video_frames = new_video_frames

    # Super-resolution 2 (29x256x160 -> 29x576x320)
    video_frames = [Image.fromarray(frame).resize((576, 320)) for frame in tensor2vid(video_frames.clone())]
    video_frames = pipe_sr_2(
        prompt,
        negative_prompt=negative_prompt,
        video=video_frames,
        strength=0.8,
        num_inference_steps=50,
    ).frames

    video_path = export_to_video(video_frames, f"{output_dir}/{prompt[:200]}.mp4")
    print(video_path)
    return video_path

css = """
#col-container {max-width: 510px; margin-left: auto; margin-right: auto;}
a {text-decoration-line: underline; font-weight: 600;}
.animate-spin {
  animation: spin 1s linear infinite;
}

@keyframes spin {
  from {
      transform: rotate(0deg);
  }
  to {
      transform: rotate(360deg);
  }
}

#share-btn-container {
  display: flex; 
  padding-left: 0.5rem !important; 
  padding-right: 0.5rem !important; 
  background-color: #000000; 
  justify-content: center; 
  align-items: center; 
  border-radius: 9999px !important; 
  max-width: 15rem;
  height: 36px;
}

div#share-btn-container > div {
    flex-direction: row;
    background: black;
    align-items: center;
}

#share-btn-container:hover {
  background-color: #060606;
}

#share-btn {
  all: initial; 
  color: #ffffff;
  font-weight: 600; 
  cursor:pointer; 
  font-family: 'IBM Plex Sans', sans-serif; 
  margin-left: 0.5rem !important; 
  padding-top: 0.5rem !important; 
  padding-bottom: 0.5rem !important;
  right:0;
}

#share-btn * {
  all: unset;
}

#share-btn-container div:nth-child(-n+2){
  width: auto !important;
  min-height: 0px !important;
}

#share-btn-container .wrap {
  display: none !important;
}

#share-btn-container.hidden {
  display: none!important;
}
img[src*='#center'] { 
    display: inline-block;
    margin: unset;
}

.footer {
        margin-bottom: 45px;
        margin-top: 10px;
        text-align: center;
        border-bottom: 1px solid #e5e5e5;
    }
    .footer>p {
        font-size: .8rem;
        display: inline-block;
        padding: 0 10px;
        transform: translateY(10px);
        background: white;
    }
    .dark .footer {
        border-color: #303030;
    }
    .dark .footer>p {
        background: #0b0f19;
    }
"""

with gr.Blocks(css=css) as demo:
    with gr.Column(elem_id="col-container"):
        gr.Markdown(
            """
            <h1 style="text-align: center;">Show-1 Text-to-Video</h1>
            <p style="text-align: center;">
            A text-to-video generation model that marries the strength and alleviates the weakness of pixel-based and latent-based VDMs. <br />
            </p>

            <p style="text-align: center;">
                <a href="https://arxiv.org/abs/2309.15818" target="_blank">Paper</a> |  
                <a href="https://showlab.github.io/Show-1" target="_blank">Project Page</a> | 
                <a href="https://github.com/showlab/Show-1" target="_blank">Github</a>
            </p>
            
            """
        )

        prompt_in = gr.Textbox(label="Prompt", placeholder="A panda taking a selfie", elem_id="prompt-in")
        #neg_prompt = gr.Textbox(label="Negative prompt", value="text, watermark, copyright, blurry, nsfw", elem_id="neg-prompt-in")
        #inference_steps = gr.Slider(label="Inference Steps", minimum=10, maximum=100, step=1, value=40, interactive=False)
        submit_btn = gr.Button("Submit")
        video_result = gr.Video(label="Video Output", elem_id="video-output")

        gr.HTML("""
            <div class="footer">
                <p>
                Demo adapted from <a href="https://huggingface.co/spaces/fffiloni/zeroscope" target="_blank">zeroscope</a> 
                by 🤗 <a href="https://twitter.com/fffiloni" target="_blank">Sylvain Filoni</a>
                </p>
            </div>
        """)
        
    submit_btn.click(fn=infer,
                    inputs=[prompt_in],
                    outputs=[video_result],
                    api_name="show-1")
    
demo.queue(max_size=12).launch(show_api=True)
