# interface_test
First, build a container with 

```
docker build -t interface_test .
```

then run the command below inside the project folder:

```
docker run -it --rm interface_test
```

The available simulations and their options will be printed, so you can run a simulation with e.g.

```
docker run -it --rm interface_test test
```

You might have to change permission for the volume that will be mounted (e.g. `chmod -R +2 brian2-sims`), as some simulations generate and save files. A more convinient approach is to work with a docker volume (see `docker volume --help`).

The docker images were tagged and pushed to dockerhub, so you can also pull the image with `docker pull pabloabur/interface_test` and run this image as shown above. Note that --gpu and --entrypoint flags can be used for using GPUs and probing the container, respectively. However, running with GPU support might require nvidia-container-toolkit. Similarly, you can use singularity to pull the image from dockerhub and run it (e.g. `singularity run --bind $(pwd)/sim_data:/app/sim_data --nv app_latest.sif test`).
