# Synthetic Unanswerable Questions in VQA

CS395T GNLP Project by Jay Liao and Ryo Kamoi

## Installation on UT Condor
*With the courtesy of [Bill Yang](https://github.com/billyang98) and his setup information on [GitHub](https://github.com/billyang98/UNITER). The following information, instruction, scripts, and code are all based on his repository's information.*

One way to download images on UT is to use the `singularity` package that's already installed in `/lusr`. To start off, make sure the following `PATH` is defined in your `.bashrc` or `.profile` to be able to load `singularity`:

```
PATH=${PATH}:/lusr/opt/singularity-3.2.1/bin
```

Restart your `bash` and run `singularity` to check. Then, run the following script with your CS login to build Chen's Docker image:

```
./scripts/condor/set_up_singularity.sh <your cs login>
```

By default, running the script will build the 2 images, one UNITER and one that UNITER use to preprocess images. These are saved to `/scratch/cluster/<your cs login>/uniter` and `/scratch/cluster/<your cs login>/uniter_butd` respectively. Then, we can start a container with:

```
singularity exec -B <UNITER REPO PATH>:/mnt/UNITER,<DATA PATH>:/mnt/data -w /scratch/cluster/<your cs login>/uniter bash
```

Notes:
- `-B` binds paths in your image when you run it.
- `-w` makes the image writeable.

## Download and Preprocess

To get the datasets used for VizWiz and UNITER's pre-trained `base` and `large`, simply run:
```
./scripts/download.sh
```

Then, we can use the images we downloaded to preprocess the datasets by running:
```
./scripts/condor/condorizer -j unans_preprocess --highgpu -- ./scripts/condor/prepro_vizwiz.sh <your cs login>
```

Some things to note during preprocessing:
- Occasionally, the GPU will run out of memory during preprocessing because the model and code is old. On encountering `Check failed: error == cudaSuccess (2 vs. 0)  out of memory`, just rerun the script. The code will skip the preprocessing that's already done.
- When encountering `Check failed: error == cudaSuccess (9 vs. 0)  invalid configuration argument`, this may be due to an image having not enough bytes (which is a useless image). This happened with `VizWiz_train_00022628.jpg` and no question used it, so the image was just removed.

## TODOs
- Verify unanswerable loss weight is correct and correspond to correct positions in BCELoss.
- Download VQA 2.0 and implement random swapping, then pre-process.
- Download QPRE and pre-process.

## Datasets
- Danna Gurari, Qing Li, Abigale J. Stangl, Anhong Guo, Chi Lin, Kristen Grauman, Jiebo Luo, and Jeffrey P. Bigham. [VizWiz Grand Challenge: Answering Visual Questions from Blind People.](https://arxiv.org/abs/1802.08218) IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2018.
- Aroma Mahendru, Viraj Prabhu, Akrit Mohapatra, Dhruv Batra, Stefan Lee. [The Promise of Premise: Harnessing Question Premises in Visual Question Answering.](https://arxiv.org/abs/1705.00601) EMNLP 2017. 
