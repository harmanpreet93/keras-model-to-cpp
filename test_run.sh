#!/bin/bash

echo 'Test for NN dumping'

# Parameters
INPUT_ARCH="arch.json"
INPUT_WEIGHTS="weights.h5"
DUMPED_NN="dumped_nn.dat"
TEST_INPUT="input.dat"
TEST_BIN="test_bin"

echo ''
echo 'Step 1'
echo 'Save architecture and weights'
python3 save_model.py -a $INPUT_ARCH -w $INPUT_WEIGHTS

echo ''
echo 'Step 2'
echo 'Save overall architecture dump in ' $DUMPED_NN
python3 dump_to_cpp.py -a $INPUT_ARCH -w $INPUT_WEIGHTS -o $DUMPED_NN

echo ''
echo 'Step 3'
echo 'Test input data using keras predict.'
python3 test_keras.py -a $INPUT_ARCH -w $INPUT_WEIGHTS -i $TEST_INPUT

echo ''
echo 'Step 4'
echo 'Test input data on the saved model using C++'
g++ test_main.cpp predict.cpp -o $TEST_BIN
./$TEST_BIN $DUMPED_NN $TEST_INPUT

echo ''
echo 'Compare outputs from step 3 and 4'

echo ''
# Clean
echo 'Cleaning after test'
rm $INPUT_ARCH
rm $INPUT_WEIGHTS
rm $DUMPED_NN
