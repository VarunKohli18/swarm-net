# Swarm-Net
This repository contains the firmware, dataset, and attestation codes used in our paper `Swarm-Net: Firmware Attestation in IoT Swarms using Graph Neural Networks and Volatile Memory` submitted to the IEEE IoT Journal, which is currently available as a [preprint on arXiv](https://arxiv.org/abs/2408.05680). Our published dataset is available on [IEEE Dataport](https://dx.doi.org/10.21227/gmee-vj41). 

If you would like to use this work, kindly cite our preprint and dataset using the following BibTeX:

`Preprint`

```\
@misc{kohli2024swarmnetfirmwareattestationiot,
      title={Swarm-Net: Firmware Attestation in IoT Swarms using Graph Neural Networks and Volatile Memory},
      author={Varun Kohli and Bhavya Kohli and Muhammad Naveed Aman and Biplab Sikdar},
      year={2024},
      eprint={2408.05680},
      archivePrefix={arXiv},
      primaryClass={cs.CR},
      url={https://arxiv.org/abs/2408.05680}, 
}
```

`Dataset`

```\
@data{gmee-vj41-24,
      doi = {10.21227/gmee-vj41},
      url = {https://dx.doi.org/10.21227/gmee-vj41},
      author = {Kohli, Varun and Kohli, Bhavya and Naveed Aman, Muhammad and Sikdar, Biplab},
      publisher = {IEEE Dataport},
      title = {IoT Swarm SRAM Dataset for Firmware Attestation},
      year = {2024}
}
```

## Firmware
The IoT device firmware for `swarm-1` and `swarm-2` are available in `./firmware`

## Memory Access
`read_memory.ipynb` and `library.py` are used for data collection. They require a physical IoT swarm setup to run. 

## Data
The compiled dataset is stored in `./data/swarm-sram-data.zip` and available on IEEE Dataport. Extract `swarm-1.pkl` and `swarm-2.pkl` from `swarm-sram-data.zip` into `./data` before running `./src/train.py`. Kindly refer to our preprint for more information on the dataset and its contents.

## Attestation

The Swarm-Net script, `./src/train.py`, trains the selected GNN architecture and saves its results on data from `swarm-1` and `swarm-2` into an Excel sheet.

Example script usage: `python train.py --dataset_name swarm-1 --device 0 --epochs 200 --latent 64 --dropout 0.5 --lr 5e-3 --noise 0.4 --runs 20 --layer_type TransformerConv`



