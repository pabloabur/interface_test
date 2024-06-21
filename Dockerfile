FROM pytorch/pytorch:2.3.0-cuda12.1-cudnn8-devel

COPY . /app/
RUN apt update && apt install -y cmake g++ curl && apt clean \
    && curl -Ls https://micro.mamba.pm/api/micromamba/linux-64/latest \
    | tar -xvj /bin/micromamba \
    && micromamba install -y -n base -f /app/env.yaml -r /opt/micromamba \
    && micromamba clean --all --yes \
    && chmod -R o+rX /opt/micromamba \
    && micromamba shell init --shell bash -r /opt/micromamba

WORKDIR /app
ENTRYPOINT ["micromamba", "run", "-n", "base", "-r", "/opt/micromamba", "python", "test.py"]
CMD ["-h"]
#FROM mambaorg/micromamba:1.5.8 as micromamba
#
## This is the image we are going add micromaba to:
#FROM pytorch/pytorch:2.3.0-cuda12.1-cudnn8-devel
#
#USER root
#
## if your image defaults to a non-root user, then you may want to make the
## next 3 ARG commands match the values in your image. You can get the values
## by running: docker run --rm -it my/image id -a
#ARG MAMBA_USER=mambauser
#ARG MAMBA_USER_ID=57439
#ARG MAMBA_USER_GID=57439
#ENV MAMBA_USER=$MAMBA_USER
#ENV MAMBA_ROOT_PREFIX="/opt/conda"
#ENV MAMBA_EXE="/bin/micromamba"
#
#COPY --from=micromamba "$MAMBA_EXE" "$MAMBA_EXE"
#COPY --from=micromamba /usr/local/bin/_activate_current_env.sh /usr/local/bin/_activate_current_env.sh
#COPY --from=micromamba /usr/local/bin/_dockerfile_shell.sh /usr/local/bin/_dockerfile_shell.sh
#COPY --from=micromamba /usr/local/bin/_entrypoint.sh /usr/local/bin/_entrypoint.sh
#COPY --from=micromamba /usr/local/bin/_dockerfile_initialize_user_accounts.sh /usr/local/bin/_dockerfile_initialize_user_accounts.sh
#COPY --from=micromamba /usr/local/bin/_dockerfile_setup_root_prefix.sh /usr/local/bin/_dockerfile_setup_root_prefix.sh
#
#RUN /usr/local/bin/_dockerfile_initialize_user_accounts.sh && \
#    /usr/local/bin/_dockerfile_setup_root_prefix.sh
#
#USER $MAMBA_USER
#
#SHELL ["/usr/local/bin/_dockerfile_shell.sh"]
#
#ENTRYPOINT ["/usr/local/bin/_entrypoint.sh"]
## Optional: if you want to customize the ENTRYPOINT and have a conda
## environment activated, then do this:
## ENTRYPOINT ["/usr/local/bin/_entrypoint.sh", "my_entrypoint_program"]
#
## You can modify the CMD statement as needed....
#CMD ["/bin/bash"]
#
## Optional: you can now populate a conda environment:
#COPY --chown=$MAMBA_USER:$MAMBA_USER env.yaml /tmp/env.yaml
#COPY --chown=$MAMBA_USER:$MAMBA_USER . /workspace
#RUN micromamba install --yes --name base -f /tmp/env.yaml && \
#    micromamba clean --all --yes
