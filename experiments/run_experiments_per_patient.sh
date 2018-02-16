# Run all experiments, sequential by patient
#patient_array=('data/Caso 1' 'data/Caso 2' 'data/Caso 3' 'data/NI 1' 'data/NI 2' 'data/NI 3')
patient_array=('data/Caso 1')
patient_names=('Caso1')

cd ..

for ((i=0;i<${#patient_array[@]};++i))
do
	echo "Running experiment for ${patient_array[i]}"
	sh experiments/patient_experiment.sh "${patient_array[i]}" "${patient_names[i]}"
done