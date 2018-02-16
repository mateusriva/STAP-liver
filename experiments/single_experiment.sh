# Run a single experiment for a given window ($1), components($2) and patient(s) ($3)
# Stores the model with a given name ('models/$4.pkl'), saves the log ('logs/$4.log')
window=$1
components=$2
patient=$3

python -u lss_train.py -v -w $window -c $components -m models/$4.pkl "$patient" > logs/$4.log