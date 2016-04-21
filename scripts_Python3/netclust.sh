#!/bin/bash

score=0
for file in $@; do
   echo Processing $file...
   netindex $file && netclust $file F1 S $score
done;

