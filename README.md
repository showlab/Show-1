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

### [Project Page](https://showlab.github.io/Show-1) | [arXiv](https://arxiv.org/abs/2309.15818) | [PDF](https://arxiv.org/abs/2309.15818) | [Replicate Demo](https://replicate.com/cjwbw/show-1) 


## News
- [10/12/2023] Code and weights released!

## Setup

### Requirements

```shell
pip install -r requirements.txt
```

Pytorch 2.0+ is highly recommended for more efficiency and speed on GPUs. 


### Weights

All weights are available in show lab [huggingface](https://huggingface.co/showlab)! Please check [key frames generation](https://huggingface.co/showlab/show-1-base), [interpolation](https://huggingface.co/showlab/show-1-interpolation), [superresolution stage 1](https://huggingface.co/showlab/show-1-sr1) and [superresolution stage 2](https://huggingface.co/showlab/show-1-sr2) modules. We also use [deep-floyd-if superresolution stage 1](https://huggingface.co/DeepFloyd/IF-II-L-v1.0) model for the first frame superresolution. To download deep-floyd-if models, you need follow their [official instructions.](https://huggingface.co/DeepFloyd/IF-II-L-v1.0)
## Inference 

To run diffusion models for text-to-video generation, run this command:

```bash
python run_inference.py
```

The output videos from different modules will be stored in "outputs" folder with the gif format. The code will automatically download  module weights from huggingface. Otherwise, you can download weights manually with git lfs then change the "pretrained_model_path" to your local path. Take key frames generation module for example:

```bash
git lfs install
git clone https://huggingface.co/showlab/show-1-base
```



## Demo Video
https://github.com/showlab/Show-1/assets/55792387/32242135-25a5-4757-b494-91bf314581e8


## Citation
If you make use of our work, please cite our paper.
```bibtex
@misc{zhang2023show1,
      title={Show-1: Marrying Pixel and Latent Diffusion Models for Text-to-Video Generation}, 
      author={David Junhao Zhang and Jay Zhangjie Wu and Jia-Wei Liu and Rui Zhao and Lingmin Ran and Yuchao Gu and Difei Gao and Mike Zheng Shou},
      year={2023},
      eprint={2309.15818},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```
## Notes
Thank you for considering our model for commercial use case. We are working with university(NUS) to figure out the exact paperwork needed for approving commercial use request. In the meantime, to speed up the process, we'd like to solicit intent of interest from community and later on we will process these requests with high priority. If you are keen, can you kindly email us at mike.zheng.shou@gmail.com and junhao.zhang@u.nus.edu to answer the following questions, if possible:
- Who are you / your company?
- What is your product / application?
- How Show-1 can benefit your product?

## Shoutouts

- This code heavily builds on [diffusers](https://github.com/huggingface/diffusers), [deep-floyd-if](https://github.com/deep-floyd/IF), [modelscope](https://huggingface.co/damo-vilab/modelscope-damo-text-to-video-synthesis), [zeroscope](https://huggingface.co/cerspense/zeroscope_v2_576w). Thanks for open-sourcing!

