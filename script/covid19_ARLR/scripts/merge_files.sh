#!/bin/bash

mkdir -p $2
fl=$(find $1 -name '*.csv' | head -n 1)
flall=$(find $1 -name '*.csv')
awk 'FNR==1' $fl > $2/merged.csv
awk 'FNR>1' $flall >> $2/merged.csv
