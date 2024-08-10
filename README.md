## Swarm-Net
This repository contains the firmware, dataset, and attestation codes used in our paper `Swarm-Net: Firmware Attestation in IoT Swarms using Graph Neural Networks and Volatile Memory` submitted to the IEEE IoT Journal. Our dataset [`IoT Swarm SRAM Dataset for Firmware Attestation`](https://dx.doi.org/10.21227/gmee-vj41) is available on IEEE Dataport. 

If you would like to use this work, kindly cite our preprint and dataset using their respective BibTeX:

`Preprint`

`Dataset`

# Firmware
The IoT device firmware for swarm-1 and swarm-2 are available in ./firmware

# Memory Access
`read_memory.ipynb` and `library.py` were used for data collection. They require a physical IoT swarm setup to run. 

# Data
The compiled dataset is stored in ./data/swarm-sram-data.zip and available on IEEE Dataport.\
Before running the attestation code, extract `swarm-1.pkl` and `swarm-2.pkl` from `swarm-sram-data.zip` into ./data/. 

# Attestation



