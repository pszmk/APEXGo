A Docker image with the requirements is provided at ```yimengzeng/apexgo:v1```, the Dockerfile is also provided. To run optimization, first start the docker image with the following command
```shell
docker run -it -v ~/APEXGo/optimization/:/workspace/ --gpus 'device=0' yimengzeng/apexgo:v1
```
then navigate to the ```constrained_bo_scripts``` folder and run ```bash optimize_gramnegative_only.sh``` for an example of optimizing template 0 for gramnegative bacteria only, with a similarity of at least 75% to the template, and producing 20 different optimized peptides at the end.