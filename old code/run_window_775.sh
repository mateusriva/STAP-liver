# Experiments for window of size 7,7,5 (245 elements)
window="775"  
components_array=(10 20 40 80 160)

for components in "${components_array[@]}"
do
	python svm_liver_hard_samples.py $window $components 100 0.95 600 0.1 > logs/split_${window}_${components}.txt
	echo "$components done"
done