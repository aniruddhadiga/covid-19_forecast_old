## Create output dir
mkdir output
mkdir output/model
mkdir output/prediciton
mkdir output/figure

## Copy slurm commands
cd output
cp ../script/run.sh .
cp ../script/run.slurm .

## Generate cfg 
cd output
python ../script/write_cfg.py

## Realtime train and predict
cd output/prediction
../run.sh
# python ../../script/lstm_ensemble.py -r 'Virginia'



