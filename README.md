# billiard
A chaotic billiard coded in Julia and enhanced with CUDA

##Prerequisites

###HDF5

HFD5 files are used for saving data, so make sure your system has HDF5

To install the Julia HDF5 package:
```
julia> Pkd.add("HDF5")
```
###CUDA
If you want to use the CUDA version you need to install CUDA in your system. For instructions on how to install CUDA refer to my phyGPU repository  

To install the Julia CUDA package:
```
julia> Pkd.add("CUDA")
```

##Usage
To run the billiard code in default ( non-CUDA ) mode:
```
$ julia billiard.jl
```
