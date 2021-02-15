#!/usr/bin/bash

fasttext supervised -input data/sst/sst_train.txt -output model_hyperopt \
-autotune-validation data/sst/sst_dev.txt -autotune-modelsize 10M -verbose 3
