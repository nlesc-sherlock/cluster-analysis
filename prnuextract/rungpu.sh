#!/bin/sh
java -Xmx16g -Djava.library.path=lib/jcuda-0.5.5/bin/lib/ -cp jar/prnuextract.jar:lib/jcuda-0.5.5/jcuda-0.5.5.jar:lib/jcuda-0.5.5/jcufft-0.5.5.jar:lib/commons-io-2.4/commons-io-2.4.jar nl.minvenj.nfi.prnu.PrnuExtractGPU
