# Runs a series of experiments for a single window ($1)
patient_array=('data/Caso\ 1' 'data/Caso\ 2' 'data/Caso\ 3' 'data/NI\ 1' 'data/NI\ 2' 'data/NI\ 3')
component_array=(10 20 40 80)

for patient in "${patient_array[@]}"
do
	for component in "${component_array[@]}"; do
		sh single_experiment.sh $1 $component $patient ${patient}_window$1_component${component} &
	done
done
