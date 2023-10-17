# Prediction interface for Cog ⚙️
# https://github.com/replicate/cog/blob/main/docs/python.md

import os
import imageio
from PIL import Image
import torch
from diffusers import (
    IFSuperResolutionPipeline,
    DiffusionPipeline,
    VideoToVideoSDPipeline,
)
from diffusers.utils.torch_utils import randn_tensor
from cog import BasePredictor, Input, Path

from showone.pipelines import (
    TextToVideoIFPipeline,
    TextToVideoIFInterpPipeline,
    TextToVideoIFSuperResolutionPipeline,
)
from showone.pipelines.pipeline_t2v_base_pixel import tensor2vid
from showone.pipelines.pipeline_t2v_sr_pixel_cond import (
    TextToVideoIFSuperResolutionPipeline_Cond,
)


class Predictor(BasePredictor):
    def setup(self) -> None:
        """Load the model into memory to make running multiple predictions efficient"""

        cache_dir = "model_cache"

        # base model
        # you can also chose "showlab/show-1-base-0.0" with more inference steps(e.g., 100) and larger gudiance scale(e.g., 12.0)
        pretrained_model_path_base = "showlab/show-1-base"
        self.pipe_base = TextToVideoIFPipeline.from_pretrained(
            pretrained_model_path_base,
            cache_dir=cache_dir,
            torch_dtype=torch.float16,
        ).to("cuda")
        self.pipe_base.enable_model_cpu_offload()
        print("Base model loaded.")

        # interpolation model 1
        pretrained_model_path_interpolation = "showlab/show-1-interpolation"
        self.pipe_interp_1 = TextToVideoIFInterpPipeline.from_pretrained(
            pretrained_model_path_interpolation,
            cache_dir=cache_dir,
            torch_dtype=torch.float16,
        ).to("cuda")
        self.pipe_interp_1.enable_model_cpu_offload()
        print("Interpolation model loaded.")

        pretrained_model_path = "DeepFloyd/IF-II-L-v1.0"
        self.pipe_sr_1_image = IFSuperResolutionPipeline.from_pretrained(
            pretrained_model_path,
            cache_dir=cache_dir,
            local_files_only=True,
            text_encoder=None,
            torch_dtype=torch.float16,
        ).to("cuda")
        self.pipe_sr_1_image.enable_model_cpu_offload()
        print("DeepFloyd model loaded.")

        # sr1
        self.pipe_sr_1_cond = TextToVideoIFSuperResolutionPipeline_Cond.from_pretrained(
            "showlab/show-1-sr1",
            cache_dir=cache_dir,
            torch_dtype=torch.float16,
        ).to("cuda")
        self.pipe_sr_1_cond.enable_model_cpu_offload()
        print("sr1 model loaded.")

        # sr2
        self.pipe_sr_2 = VideoToVideoSDPipeline.from_pretrained(
            "showlab/show-1-sr2",
            cache_dir=cache_dir,
            torch_dtype=torch.float16,
        ).to("cuda")
        self.pipe_sr_2.enable_model_cpu_offload()
        self.pipe_sr_2.enable_vae_slicing()
        print("sr2 model loaded.")

    def predict(
        self,
        prompt: str = Input(description="Input text for video generation."),
        negative_prompt: str = Input(
            description="Content you do not want to see in the output.",
            default="low resolution, blur",
        ),
        num_frames: int = Input(
            description="Number of frames in the generated video", default=8
        ),
        num_base_steps: int = Input(
            description="Number of denoising steps in key frame generation", default=75
        ),
        num_interpolation_steps: int = Input(
            description="Number of denoising steps in interpolation", default=75
        ),
        num_sr1_steps: int = Input(
            description="Number of denoising steps in stage 1 super resolution",
            default=175,
        ),
        num_sr2_steps: int = Input(
            description="Number of denoising steps in stage 2 super resolution",
            default=50,
        ),
        seed: int = Input(
            description="Random seed. Leave blank to randomize the seed", default=None
        ),
    ) -> Path:
        """Run a single prediction on the model"""
        if seed is None:
            seed = int.from_bytes(os.urandom(2), "big")
        print(f"Using seed: {seed}")

        # text embeds
        prompt_embeds, negative_embeds = self.pipe_base.encode_prompt(prompt)

        # keyframes generation
        video_frames = self.pipe_base(
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_embeds,
            num_frames=num_frames,
            height=40,
            width=64,
            num_inference_steps=num_base_steps,
            guidance_scale=9.0,
            generator=torch.manual_seed(seed),
            output_type="pt",
        ).frames
        # interpolation (2fps -> 7.5fps)
        bsz, channel, num_frames, height, width = video_frames.shape
        new_num_frames = 3 * (num_frames - 1) + num_frames
        new_video_frames = torch.zeros(
            (bsz, channel, new_num_frames, height, width),
            dtype=video_frames.dtype,
            device=video_frames.device,
        )
        new_video_frames[:, :, torch.arange(0, new_num_frames, 4), ...] = video_frames

        init_noise = randn_tensor(
            (bsz, channel, 5, height, width),
            generator=torch.manual_seed(seed),
            device=video_frames.device,
            dtype=video_frames.dtype,
        )

        for i in range(num_frames - 1):
            batch_i = torch.zeros(
                (bsz, channel, 5, height, width),
                dtype=video_frames.dtype,
                device=video_frames.device,
            )
            batch_i[:, :, 0, ...] = video_frames[:, :, i, ...]
            batch_i[:, :, -1, ...] = video_frames[:, :, i + 1, ...]
            batch_i = self.pipe_interp_1(
                pixel_values=batch_i,
                prompt_embeds=prompt_embeds,
                negative_prompt_embeds=negative_embeds,
                num_frames=batch_i.shape[2],
                height=40,
                width=64,
                num_inference_steps=num_interpolation_steps,
                guidance_scale=4.0,
                generator=torch.manual_seed(seed),
                output_type="pt",
                init_noise=init_noise,
                cond_interpolation=True,
            ).frames

            new_video_frames[:, :, i * 4 : i * 4 + 5, ...] = batch_i

        video_frames = new_video_frames

        # sr1
        bsz, channel, num_frames, height, width = video_frames.shape
        window_size, stride = 8, 7
        new_video_frames = torch.zeros(
            (bsz, channel, num_frames, height * 4, width * 4),
            dtype=video_frames.dtype,
            device=video_frames.device,
        )
        for i in range(0, num_frames - window_size + 1, stride):
            batch_i = video_frames[:, :, i : i + window_size, ...]

            if i == 0:
                first_frame_cond = self.pipe_sr_1_image(
                    image=video_frames[:, :, 0, ...],
                    prompt_embeds=prompt_embeds,
                    negative_prompt_embeds=negative_embeds,
                    height=height * 4,
                    width=width * 4,
                    num_inference_steps=70,
                    guidance_scale=4.0,
                    noise_level=150,
                    generator=torch.manual_seed(seed),
                    output_type="pt",
                ).images
                first_frame_cond = first_frame_cond.unsqueeze(2)
            else:
                first_frame_cond = new_video_frames[:, :, i : i + 1, ...]

            batch_i = self.pipe_sr_1_cond(
                image=batch_i,
                prompt_embeds=prompt_embeds,
                negative_prompt_embeds=negative_embeds,
                first_frame_cond=first_frame_cond,
                height=height * 4,
                width=width * 4,
                num_inference_steps=num_sr1_steps,
                guidance_scale=7.0,
                noise_level=250,
                generator=torch.manual_seed(seed),
                output_type="pt",
            ).frames
            new_video_frames[:, :, i : i + window_size, ...] = batch_i

        video_frames = new_video_frames

        # sr2
        video_frames = [
            Image.fromarray(frame).resize((576, 320))
            for frame in tensor2vid(video_frames.clone())
        ]
        video_frames = self.pipe_sr_2(
            prompt,
            negative_prompt=negative_prompt,
            video=video_frames,
            strength=0.8,
            num_inference_steps=num_sr2_steps,
            generator=torch.manual_seed(seed),
            output_type="pt",
        ).frames

        out_path = "/tmp/out.mp4"
        imageio.mimsave(out_path, tensor2vid(video_frames.clone()), fps=8)
        return Path(out_path)
