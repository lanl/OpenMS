#!/bin/sh

pwd=$PWD

export CC=gcc
export CCFLAG="-c -fPIC -O2 -I/home/zhy/local/gsl-2.4/include -I/home/zhy/anaconda3/include/python3.8/ -I$pwd"
export OBJECTS="incoherent.o metalobj.o memory.o parameter.o input.o sigma.o timeupdate.o source.o output.o farfield.o energy.o transmap.o pFDTD_wrap.o"

echo $CC

$CC  $CCFLAG  incoherent.c
$CC  $CCFLAG  metalobj.c
$CC  $CCFLAG  memory.c
$CC  $CCFLAG  parameter.c
$CC  $CCFLAG  input.c
$CC  $CCFLAG  sigma.c
$CC  $CCFLAG  timeupdate.c
$CC  $CCFLAG  source.c
$CC  $CCFLAG  output.c
$CC  $CCFLAG  farfield.c
$CC  $CCFLAG  energy.c
$CC  $CCFLAG  transmap.c

swig -python pFDTD.i
$CC  $CCFLAG  pFDTD_wrap.c

ld -shared $OBJECTS -o _fdtdc.so  -L/home/zhy/local/gsl-2.4/lib -lgsl -lgslcblas





