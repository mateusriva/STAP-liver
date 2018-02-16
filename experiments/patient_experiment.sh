# Runs a series of experiments for a single patient folder ($1), named as $2
window_array=('333' '553' '775')
component_array=(10 20 40 80)

for window in "${window_array[@]}"
do
	for component in "${component_array[@]}"; do
		sh experiments/single_experiment.sh $window $component "$1" $2_window${window}_component${component} &
	done
done
