#!/bin/sh
java -Xmx60g -Xms58g -Djava.library.path=lib/jcuda-0.8.0/lib/ -cp jar/prnuextract.jar:lib/jcuda-0.8.0/jcuda-0.8.0.jar:lib/jcuda-0.8.0/jcufft-0.8.0.jar:lib/commons-io-2.4/commons-io-2.4.jar nl.minvenj.nfi.prnu.PrnuExtractGPU $@
