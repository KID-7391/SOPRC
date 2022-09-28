# SOPRC
An implement of the NeurIPS 2022 paper: [**Exploring the Algorithm-Dependent Generalization of AUPRC Optimization with List Stability**](https://arxiv.org/abs/2209.13262).

## Environments
* **Ubuntu** 16.04
* **CUDA** 11.1
* **Python** 3.8.10
* **Pytorch** 1.8.2+cu11

See `requirement.txt` for others.

## Data preparation

Download [SOP](https://drive.google.com/uc?export=download&id=1TclrpQOF_ullUP99wk_gjGN8pKvtErG8), [PKU VehicleID](https://www.pkuml.org/resources/pku-vehicleid.html), and [iNaturalist](https://github.com/visipedia/inat_comp/tree/master/2018#Data). Unzip these files and place then in `./data/[dataset]/images`.

## Training

Download the pretrained model of [ResNet-50](https://download.pytorch.org/models/resnet50-0676ba61.pth) in `./pretrained_models`.

Run the following command for training & validation

```shell
bash scripts/run.sh config/$DATASET/$CONFIG $gpu_id
```

For example,
```shell
bash scripts/run.sh config/iNaturalist/soprc_sgd.yaml 0
```

## Evaluation

```shell
bash scripts/test.sh config/$DATASET/$CONFIG $gpu_id
```

For example,
```shell
bash scripts/test.sh config/iNaturalist/soprc_sgd.yaml 0
```

## Losses

The following methods are provided in this repository (see Appendix in our paper):

* Pairwise Losses, including Contrastive Loss, Triplet Loss, Multi-Similarity (MS) Loss, Cross-Batch Memory (XBM)
* Ranking-Based Losses, including SmoothAP, FastAP, DIR, BlackBox, Area Under the ROC Curve Loss (AUROC), and SOPRC (Ours)

See `losses/loss_warpper.py` for usage.

By default, these losses take a dict with keys "feat" and "target" as input.
Here the feature is an $(N\times M) \times D$ tensor, where $N$ is the number of ids, $M$ is the number of positive examples for each id and $D$ is feature dimension.
The target is an $(N\times M) \times 1$ tensor, where the first $M$ examples belong to the same id, and so on.
See `config/demo.yaml` for more details on configures. For example, by setting `batchsize = 224`, `num_sample_per_id = 4`, `output_channels = 512`, we have $N = 56, M = 4, D = 512$, and the target could be $[2,2,2,2,1,1,1,1,4,4,4,4,...]$.

## References
If this code is helpful to you, please consider citing our paper:
```
@inproceedings{wen2022exploring,
  title={Exploring the Algorithm-Dependent Generalization of AUPRC Optimization with List Stability},
  author={Wen, Peisong and Xu, Qianqian and Yang, Zhiyong and He, Yuan and Huang, Qingming},
  booktitle={Annual Conference on Neural Information Processing Systems},
  year={2022}
}
```
