# Experiments for window of size 5,5,3 (75 elements)
window="553"  
components_array=(10 20 40)

for components in "${components_array[@]}"
do
	python svm_liver_hard_samples.py $window $components > logs/split_${window}_${components}.txt
done