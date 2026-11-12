import pandas as pd

df_bb = pd.read_feather('srbench-master/results/black-box_results.feather')
df_gt = pd.read_feather('srbench-master/results/ground-truth_results.feather')


total_dataset_bb = [x for x in set(df_bb["dataset"]) if type(x) == str]
total_dataset_gt = [x for x in set(df_gt["dataset"]) if type(x) == str]

for i in sorted(total_dataset_bb + total_dataset_gt): print(i)


print(len(total_dataset_bb))
print(len(total_dataset_gt))