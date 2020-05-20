1.perceptron

1>.To run ERM on linearly-separable-dataset.csv

For example, the file data structure is the following:

HW1 file:
	code file:
		Adaboost.py
		perceptron.py
	data file:
		Breast_cancer_data.csv
		linearly-separable-dataset.csv

If the working directory is HW1 file, run:

python ./code/perceptron.py --dataset ./data/linearly-separable-dataset.csv --mode erm --max_iterations <Int>(optional,default 1000000) --learning_rate <Float>(optional, default 0.1)

Explain: dataset and mode are required. max_iterations and learning_rate are optional.

2>.To run 10-folds cross validation on linearly-separable-dataset.csv:

python ./code/perceptron.py --dataset ./data/linearly-separable-dataset.csv --mode 10-fold --max_iterations <Int>(optional,default 1000000) --learning_rate <Float>(optional, default 0.1)

3>.To run ERM on Breast_cancer_data.csv:

python ./code/perceptron.py --dataset ./data/Breast_cancer_data.csv --mode erm --max_iterations <Int>(optional,default 1000000) --learning_rate <Float>(optional, default 0.1)

4>.To run 10-folds cross validation on Breast_cancer_data.csv:

python ./code/perceptron.py --dataset ./data/Breast_cancer_data.csv --mode 10-fold --max_iterations <Int>(optional,default 1000000) --learning_rate <Float>(optional, default 0.1)

2. Adaboost

1>.To run ERM on Breast_cancer_data.csv:

python ./code/Adaboost.py --dataset ./data/Breast_cancer_data.csv --mode erm --number_of_classifier <Int>(optional, default 8)

2>.To run 10-folds cross validation on Breast_cancer_data.csv:

python ./code/Adaboost.py --dataset ./data/Breast_cancer_data.csv --mode 10-fold --number_of_classifier <Int>(optional, default 8)




