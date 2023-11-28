FROM ubuntu:22.04

ENV ENVDIR=env

# install sudo
RUN apt-get -yq update && apt-get -yq install sudo

WORKDIR /$ENVDIR

# install packages
RUN sudo apt-get install -yq git
RUN sudo apt-get install --no-install-recommends -yq make gcc gfortran libssl-dev cmake
RUN sudo apt-get install -yq libopenblas-dev libmpich-dev libblas-dev liblapack-dev libscalapack-mpi-dev libhdf5-mpich-dev
RUN sudo apt-get install -yq vim git-lfs valgrind wget astyle
RUN sudo apt-get clean -q

# download dependencies
ENV LIB_DIR=/$ENVDIR/dependencies
WORKDIR $LIB_DIR

# install python
RUN sudo apt-get update && sudo apt-get install -yq python3 python3-dev python3-pip

RUN echo "numpy" >> requirements.txt
RUN echo "scipy" >> requirements.txt
RUN echo "argparse" >> requirements.txt
RUN echo "tables" >> requirements.txt
RUN echo "PyYAML" >> requirements.txt
RUN echo "h5py" >> requirements.txt
RUN echo "pybind11" >> requirements.txt
RUN echo "pytest" >> requirements.txt
RUN echo "mpi4py" >> requirements.txt
RUN sudo pip3 install --upgrade pip
RUN sudo pip3 install -r ./requirements.txt

# install lldb and gdb for debugging
RUN sudo apt-get install -yq lldb gdb

# install julia
WORKDIR $LIB_DIR
RUN sudo wget https://julialang-s3.julialang.org/bin/linux/aarch64/1.9/julia-1.9.3-linux-aarch64.tar.gz
RUN sudo tar -zxvf julia-1.9.3-linux-aarch64.tar.gz
ENV PATH="$LIB_DIR/julia-1.9.3/bin:$PATH"

# create and switch to a user
ENV USERNAME=test
RUN echo "$USERNAME ALL=(ALL) NOPASSWD:ALL" >> /etc/sudoers
RUN useradd --no-log-init -u 1001 --create-home --shell /bin/bash $USERNAME
RUN adduser $USERNAME sudo
USER $USERNAME
WORKDIR /home/$USERNAME

# install julia packages
RUN julia -e 'using Pkg; Pkg.add("Plots")'
RUN julia -e 'using Pkg; Pkg.add("GR")'
RUN julia -e 'using Pkg; Pkg.add("FFTW")'
RUN julia -e 'using Pkg; Pkg.add("Ipopt")'
RUN julia -e 'using Pkg; Pkg.add("JLD2")'
RUN julia -e 'using Pkg; Pkg.add("LaTeXStrings")'
RUN julia -e 'using Pkg; Pkg.add("Printf")'
RUN julia -e 'using Pkg; Pkg.add("Random")'
RUN julia -e 'using Pkg; Pkg.add("SparseArrays")'
RUN julia -e 'using Pkg; Pkg.add("Test")'
RUN julia -e 'using Pkg; Pkg.add("FileIO")'
RUN julia -e 'using Pkg; Pkg.add("Suppressor")'
RUN julia -e 'using Pkg; Pkg.add("Documenter")'

## create julia config file
#WORKDIR /home/$USERNAME/.julia/config
#RUN echo ENV["PLOTS_DEFAULT_BACKEND"]="GR" >> startup.jl

WORKDIR /home/$USERNAME

