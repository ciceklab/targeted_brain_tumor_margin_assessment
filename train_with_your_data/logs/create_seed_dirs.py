import os 
import os.path as osp
import pdb

# create seed folders
leaf_folders = [root for root, dirs, files in os.walk("./") if not dirs]
seed_folders = []
seeds = [6, 81, 35]
for leaf in leaf_folders:
    for seed in seeds:
        seed_folders.append(osp.join(leaf, f"seed_{seed}"))

# THINK TWICE BEFORE UNCOMMENTING THE CODE SEGMENT BELOW
# for folder in seed_folders:
#     os.mkdir(folder)

pdb.set_trace()
