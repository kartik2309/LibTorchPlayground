#!/bin/bash

executable_path="$1"
n_procs="${2:=4}"
mpirun -n "$n_procs" "$executable_path"
