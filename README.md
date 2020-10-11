# Yet another Deep Danbooru project
But based on [RegNetY-8G](https://arxiv.org/abs/2003.13678), relative lightweight, designed to run fast on GPU. \
Training is done using mixed precision training on a single RTX2080Ti for 3 weeks. \
Some code are from https://github.com/facebookresearch/pycls
# What do I need?
You need to download [save_4000000.ckpt]() from release and place on the same folder as `test.py`.
# How to use?
`python test.py --model save_4000000.ckpt --image <PATH_TO_IMAGE>`
# What to do in the future?
1. Quantize to 8 bit

