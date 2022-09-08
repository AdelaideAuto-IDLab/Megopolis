# Megopolis
Megopolis Project Code and Documentation.

## Introduction

Megopolis is an open source framework for resampling algorithms on GPUs, which is an essential part to improve performance of Sequential Monte Carlo (SMC) or Particle Filtering algorithms.  This repository is provided as part of the following papers :

Chesser J., Nguyen H. V., & Damith C. (2021). *The Megopolis Resampler: Memory Coalesced Resampling on GPUs*, which is accepted to [Digital Signal Processing](https://www.journals.elsevier.com/digital-signal-processing/). See [Paper](https://arxiv.org/abs/2109.13504). 

Cite using:

  ```
  @article{chesser2021,
    title={Memory Coalesced Metropolis Resampling: The Megopolis Resampler},
    author={Chesser, Joshua A. and Nguyen, Hoa Van and Ranasinghe, Damith C.},
    journal={Digital Signal Processing},
    year={2021}
  }
  ```

## Structure

The implementation of the GPU accelerated resampling algorithms can be found in [`src/resampling`](src/resampling).
This code is header only and exists in the namespace `resampling`. The remaining code uses the resampling 
algorithm implmentations for benchmarking.

## Running the Benchmarks

The benchmarks rely on [RapidJson](https://github.com/Tencent/rapidjson/) for parsing the config
files. Place the [`rapidjson/include/rapidjson`](https://github.com/Tencent/rapidjson/tree/master/include/rapidjson) 
header only libary into the include directory:

```console
$ mkdir Megopolis/include
$ cp -R rapidjson/include/rapidjson Megopolis/include/rapidjson
```

Ensure you have the [CUDA Toolkit](https://docs.nvidia.com/cuda/index.html) installed. 
*NOTE* This build process has been tested with CUDA version 11.7.

Build the target directory and run the make file. *NOTE* For windows users, [CUDA can be used on WSL2](https://docs.nvidia.com/cuda/wsl-user-guide/index.html):

```console
$ cd Megopolis
$ mkdir target
$ make
```

This will create two binaries `target/resample_test` and `target/filter_bench`. `resample_test` will 
run the resampling benchmarks while `filter_bench` will run the filtering benchmarks. To run these 
benchmarks, simply pass the desired config file as a parameter:

```console
$ ./target/resample_test configs/resample_config.json
$ ./target/filter_bench configs/filter_config.json
```

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details
