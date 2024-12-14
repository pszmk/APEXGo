A Docker image with the requirements is provided at ```yimengzeng/apexgo:v1```, the Dockerfile is also provided. To run optimization, first start the docker image with the following command
```shell
docker run -it -v ~/APEXGo/optimization/:/workspace/ --gpus 'device=0' yimengzeng/apexgo:v1
```
then navigate to the ```constrained_bo_scripts``` folder and run ```bash optimize_gramnegative_only.sh``` for an example of optimizing template 0 for gramnegative bacteria only, with a similarity of at least 75% to the template, and producing 20 different optimized peptides at the end.


For a local setup using conda, run the following:
```shell
conda create --name apexgo python=3.10
conda activate apexgo
conda install pytorch torchvision torchaudio pytorch-cuda=12.4 -c pytorch -c nvidia
pip install tqdm==4.65.0 wandb==0.18.6 botorch==0.12.0 selfies==2.1.2 guacamol==0.5.5 rdkit==2024.3.6 lightning==2.4.0 joblib==1.4.2 fire==0.7.0 levenshtein==0.26.1 rotary_embedding_torch==0.8.5 gpytorch==1.13 pandas==2.2.3 numpy==1.24.3 fcd_torch==1.0.7 matplotlib==3.9.2
```

This is tested with a NVIDIA RTX A6000 with driver version 535.86.10 and CUDA driver version 12.2.