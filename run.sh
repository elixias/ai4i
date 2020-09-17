#!/bin/sh
echo "This script is written for AIAP Batch 6 Technical Assessment."
cmd="python mlp.py"
echo "Input file to perform prediction on:"
echo "(Must be in following format:)"
echo "date,hr,weather,temperature,feels-like-temperature,relative-humidity,windspeed,psi,guest-users,registered-users"
echo "Input file >"
read f_input
if [ -n  "$f_input" ]; then
cmd="$cmd -i $f_input"
fi
echo "Output file to write to? (If none is specified)"
echo "Output file >"
read f_output
if [ -n  "$f_output" ]; then
cmd="$cmd -o $f_output"
fi
echo "Which linear model will you like to use? (linear, lasso, ridge)"
echo "Linear model >"
read model
if [ -n  "$model" ]; then
cmd="$cmd --model_type $model"
fi
echo "How to handle negative values in training and input files? (raise, zero, nan)"
echo "Handle negative >"
read neg
if [ -n  "$neg" ]; then
cmd="$cmd --handle_neg $neg"
fi
echo "Test size to use for train test split? (Expect a float)"
echo "Test size >"
read testsize
if [ -n  "$testsize" ]; then
cmd="$cmd --test_size $testsize"
fi
echo "Seed to use for train test split? (Expect an int)"
echo "Seed >"
read seed
if [ -n  "$seed" ]; then
cmd="$cmd --seed $seed"
fi
echo "Target variable? (guest-users, registered-users, total)"
echo "nTarget >"
read target
if [ -n  "$target" ]; then
cmd="$cmd --target $target"
fi
echo "Use SimpleImputer in the pipeline? (0 or 1)"
echo "Imputer >"
read imputer
if [ -n  "$imputer" ]; then
cmd="$cmd --imputer $imputer"
fi
echo "Use Normalizer in the pipeline? (0 or 1)"
echo "Normalizer >"
read norm
if [ -n  "$norm" ]; then
cmd="$cmd --normalize $norm"
fi
echo "PolynomialFeatures number of degrees? (0, 1, 2)"
echo "Values higher than 2 not recommended."
echo "Degrees >"
read degree
if [ -n  "$degree" ]; then
cmd="$cmd --poly $degree"
fi
echo "The script will run with the following variables:
File input: $f_input
Output File: $f_output
Model: $model
Neg Handler: $neg
Test Size: $testsize
Seed: $seed
Target: $target
Imputer: $imputer
Norm: $norm
Degree: $degree"
echo "Proceed with these settings? (Y/N):"
read proceed
if [ "$proceed" = "y" ] || [ "$proceed" = "Y" ]; then
echo $cmd
eval $cmd
fi