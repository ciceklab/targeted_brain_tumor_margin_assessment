import os 
import os.path as osp
import pdb

# create seed folders
leaf_folders = [root for root, dirs, files in os.walk("./") if not dirs]

# THINK TWICE BEFORE UNCOMMENTING THE CODE SEGMENT BELOW
# create placeholder files at seed_folders
for folder in leaf_folders:
    with open(osp.join(folder, ".placeholder"), "w") as f:
        f.write("placeholder: delete this file when directory is filled\n")