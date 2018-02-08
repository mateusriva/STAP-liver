# Experiments for window of size 3,3,3 (27 elements)
window="333"  
components_array=(10 20)

for components in "${components_array[@]}"
do
	python svm_liver_hard_samples.py $window $components > logs/split_${window}_${components}.txt
done