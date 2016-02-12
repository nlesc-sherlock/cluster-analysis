#!/bin/sh
java -Xmx62g -Xms60g -Djava.library.path=lib/jcuda-0.7.0a/bin/lib/ -cp jar/prnuextract.jar:lib/jcuda-0.7.0a/jcuda-0.7.0a.jar:lib/jcuda-0.7.0a/jcufft-0.7.0a.jar:lib/commons-io-2.4/commons-io-2.4.jar nl.minvenj.nfi.prnu.PrnuExtractGPU $@
