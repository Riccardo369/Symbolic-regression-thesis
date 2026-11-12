# Symbolic regression thesis

The goal of the project is to propose Greedy Statistical Mapping, a  new symbolic regression algorithm, that use statistical non parametric heuristic for resolve the SR problem.

For having a complete map of files and folders:

- InformationSOTA contains all necessary files about the datasets, plus an analysis notebook used for analyzing the datasets.

- BenchmarkCompleteResults and BenchmarkResults contain json files that represent results from training phases using datasets from InformationSOTA. In BenchmarkResults
  there are json files that containts the timestamp and tree form of model met during the iterations, and BenchmarkCompleteResults adds more information for each iteration (like the number of nodes, MSE, MAE, ...).

- .env only contains the random seeds of internal experiments (non invertible and heuristic race experiments).

- BenchmarkExperiments notebook execute the DatasetDimension results, execute training phases on datasets and fill BenchmarkResults / BenchmarkCompleteResults.

- DatasetDimension.json contains information about datasets

- Exception.py contains different types of Exception (useful for implementation of logic for the project).

- Errors.py contains different metrics of errors: MSE, MAE and R2

- Indicators.py contains all non parametrical tools useful for algorithm: Pearson / Spearman / Chatterjee / Kurtosis coefficient and FOCI algorithm

- Kaggle machine information.txt contains all information about in the machine where the experiments (from BenchmarkExperiments execution) are executed

- LearningStrategy.py contains one class "GreedyStatisticalMapping", that class represents the learning algorithm proposed for SR problem

- main.py represents a test file where execute training phases (for debugging) (FORSE DA ELIMINARE ?)

- Operations.py contains all type of operation node (with all basic methods used for example in forward and backward steps).

- Model.py contains a class that represent the structure of the model. A model contains a tree composed by operation nodes in Operations.py.

- NonInvertibleFunctions.py contains methods for the pseudo inverse of non invertible functions (correlated to which type of non invertible function are in Operations.py).

- Simplifications.py contains all useful tools for apply the syntax simplification of a tree model.

- StatisticalHeuristics.py contains all necessary for apply the non parametrical statistical heuristics (used in "GreedyStatisticalMapping" algorithm and in
  heuristic race internal experiment).

- ShowTest.py contains a script where the results of internal experiments are shown.

- Every Test*.py files execute a different internal experiment.

- Every Test*.json contains the result from Test*.py experiment.

- InternalExperimentsExecution.py execute from one file all internal experiments.