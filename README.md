# billiard
A chaotic billiard coded in Julia and enhanced with CUDA.

##Prerequisites

###HDF5

HFD5 files are used for saving data, so make sure your system has HDF5 installed.

To install the Julia HDF5 package:
```
julia> Pkd.add("HDF5")
```
###CUDA
If you want to use the CUDA version you need to install CUDA in your system. For instructions on how to install CUDA refer to my phyGPU repository [HERE.](https://github.com/bvillasen/phyGPU)

To install the Julia CUDA package:
```
julia> Pkd.add("CUDA")
```

##Usage
To run the billiard code in default ( non-CUDA ) mode:
```
$ julia billiard.jl
```
You can set the main parameters from the command line when you run the billiard code to do this use the keywords:
* **n_part**:  Number of particles
* **n_snap**:  Number of snapshots
* **iter_per_snap**:  Number of iterations each particle will do per snapshot
* **time_per_snap**:  Time ( or distance ) each particle will advance to save it's position for each snapshot

For example to run with the next parameters: 1024 particles, 100 snapshots, 100 iterations per snapshot and a time per snapshot equal to 20 use:
```
$ julia billiard.jl n_part=1024 n_snap=100 iter_per_snap=100 time_per_snap=20
```

The order of the keywords makes no difference.

To run the CUDA enhanced version add the keyword **cuda**, for example:
```
$ julia billiard.jl n_part=1024 n_snap=100 iter_per_snap=100 time_per_snap=20 cuda
```

##Benchmarks
For the next set of parameters:
* **n_part** = 10240
* **n_snap** = 100
* **iter_per_snap** = 1000
* **time_per_snap** = 100
* Total iterations per particle = n_snap * iter_per_snap = 100000

An i7 Intel ( 2.2 GHz ) took  19 min  ( 1140 secs )  (single core implementation ).
An old and small laptop GPU ( GT 260M, 96 cores  ) took 16 secs.
A GTX TITAN ( 2688 cores ) took 1.2 secs.
**CONCLUSION: Get a GPU and start using it now!**
