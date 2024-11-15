<div align="center" width="100%">
<h1>🎬Show-1</h1>
</div>

<div>
<div align="center">
    <a href='https://junhaozhang98.github.io/' target='_blank'>David Junhao Zhang<sup>*</sup></a>&emsp;
    <a href='https://zhangjiewu.github.io/' target='_blank'>Jay Zhangjie Wu<sup>*</sup></a>&emsp;
    <a href='https://jia-wei-liu.github.io/' target='_blank'>Jia-Wei Liu<sup>*</sup></a>
    <br>
    <a href='https://ruizhaocv.github.io/' target='_blank'>Rui Zhao<sup></sup></a>&emsp;
    <a href='https://siacorplab.nus.edu.sg/people/ran-lingmin/' target='_blank'>Lingmin Ran<sup></sup></a>&emsp;
    <a href='https://ycgu.site/' target='_blank'>Yuchao Gu<sup></sup></a>&emsp;
    <a href='https://scholar.google.com/citations?user=No9OsocAAAAJ&hl=en' target='_blank'>Difei Gao<sup></sup></a>&emsp;
    <a href='https://sites.google.com/view/showlab/home?authuser=0' target='_blank'>Mike Zheng Shou<sup>&#x2709</sup></a>
</div>
<div>
<div align="center">
    <a href='https://sites.google.com/view/showlab/home?authuser=0' target='_blank'>Show Lab, National University of Singapore</a>
    </br>
    <sup>*</sup> Equal Contribution&emsp;
    <sup>&#x2709</sup> Corresponding Author
</div>

-----------------

![](https://img.shields.io/github/stars/showlab/Show-1?style=social)
[![Hits](https://hits.seeyoufarm.com/api/count/incr/badge.svg?url=https%3A%2F%2Fgithub.com%2Fshowlab%2FShow-1&count_bg=%2379C83D&title_bg=%23555555&icon=&icon_color=%23E7E7E7&title=hits&edge_flat=false)](https://hits.seeyoufarm.com)


### [Project Page](https://showlab.github.io/Show-1) | [arXiv](https://arxiv.org/abs/2309.15818) | [PDF](https://arxiv.org/abs/2309.15818) | [🤗 Space](https://huggingface.co/spaces/showlab/Show-1) | [Colab](https://colab.research.google.com/github/camenduru/Show-1-colab/blob/main/Show_1_steps_colab.ipynb) | [Replicate Demo](https://replicate.com/cjwbw/show-1) 


## News
- [10/06/2024] Show-1 was accepted to IJCV!
- [10/12/2023] Code and weights released!

## Setup

### Requirements

```shell
pip install -r requirements.txt
```

Note: PyTorch 2.0+ is highly recommended for more efficiency and speed on GPUs. 


### Weights

All model weights for Show-1 are available on [Show Lab's HuggingFace page](https://huggingface.co/showlab): Base Model ([show-1-base](https://huggingface.co/showlab/show-1-base)), Interpolation Model ([show-1-interpolation](https://huggingface.co/showlab/show-1-interpolation)), and Super-Resolution Model ([show-1-sr1](https://huggingface.co/showlab/show-1-sr1), [show-1-sr2](https://huggingface.co/showlab/show-1-sr2)).

Note that our [show-1-sr1](https://huggingface.co/showlab/show-1-sr1) incorporates the image super-resolution model from DeepFloyd-IF, [DeepFloyd/IF-II-L-v1.0](https://huggingface.co/DeepFloyd/IF-II-L-v1.0), to upsample the first frame of the video. To obtain the respective weights, follow their [official instructions](https://huggingface.co/DeepFloyd/IF-II-L-v1.0).

## Usage 

To generate a video from a text prompt, run the command below:

```bash
python run_inference.py
```

By default, the videos generated from each stage are saved to the `outputs` folder in the GIF format. The script will automatically fetch the necessary model weights from HuggingFace. If you prefer, you can manually download the weights using git lfs and then update the `pretrained_model_path` to point to your local directory. Here's how:

```bash
git lfs install
git clone https://huggingface.co/showlab/show-1-base 
```

A demo is also available on the [`showlab/Show-1` 🤗 Space](https://huggingface.co/spaces/showlab/Show-1).
You can use the gradio demo locally by running:

```bash
python app.py
```


## Demo Video
https://github.com/showlab/Show-1/assets/55792387/32242135-25a5-4757-b494-91bf314581e8


## Citation
If you make use of our work, please cite our paper.
```bibtex
@article{zhang2023show,
  title={Show-1: Marrying Pixel and Latent Diffusion Models for Text-to-Video Generation},
  author={Zhang, David Junhao and Wu, Jay Zhangjie and Liu, Jia-Wei and Zhao, Rui and Ran, Lingmin and Gu, Yuchao and Gao, Difei and Shou, Mike Zheng},
  journal={arXiv preprint arXiv:2309.15818},
  year={2023}
}
```

## Commercial Use

We are working with the university (NUS) to figure out the exact paperwork needed for approving commercial use request. In the meantime, to speed up the process, we'd like to solicit intent of interest from community and later on we will process these requests with high priority. If you are keen, can you kindly email us at mike.zheng.shou@gmail.com and junhao.zhang@u.nus.edu to answer the following questions, if possible:
- Who are you / your company?
- What is your product / application?
- How Show-1 can benefit your product?

## Shoutouts

- This work heavily builds on [diffusers](https://github.com/huggingface/diffusers), [deep-floyd/IF](https://github.com/deep-floyd/IF), [modelscope](https://huggingface.co/damo-vilab/modelscope-damo-text-to-video-synthesis), and [zeroscope](https://huggingface.co/cerspense/zeroscope_v2_576w). Thanks for open-sourcing!
- Thanks [@camenduru](https://github.com/camenduru) for providing the CoLab demo and [@chenxwh](https://github.com/chenxwh) for providing replicate demo.

