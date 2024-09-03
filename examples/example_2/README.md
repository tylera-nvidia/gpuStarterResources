# Example 2
This Example demonstrates a simple vector multiple, with 3 separate device-memory backed tensors A, B, and C, where `C = A .* B`

# Building
The `run.sh` script in the examples folder provides an environment capable of building all the examples. I would suggest running from this environment, however any CUDA-enabled environment should work.


Use the following commands to build the example:

`mkdir build`

`cd build`

`cmake ..`

`make -j`


# Running

Run the example with the command `./example_2`

You should see the following output:

```
I have no name!@72063c2be218:/scratch/projects/gpuStarterResources/examples/example_2/build$ make -j
Consolidate compiler generated dependencies of target example_2
[ 50%] Building CUDA object CMakeFiles/example_2.dir/example_2.cu.o
[100%] Linking CUDA executable example_2
[100%] Built target example_2
I have no name!@72063c2be218:/scratch/projects/gpuStarterResources/examples/example_2/build$ ./example_2 
Average elapsed time per iteration is: 3372.13us
```


# Profiling Commands

to profile with nsys, try the following:
`nsys profile -o example_2 ./example_2`

to profile with ncu, try the following:
`ncu --set=full --import-source=true -c 2 -f -o example_2 ./example_2`
