import datasets
import glob
from pathlib import Path
import os

datadir = "/work/arjunguha-research-group/franlucc/projects/codetrace/codetrace/type_inf_exp/exp_data2"

for p in glob.glob(f"{datadir}/*"):
    
    if os.path.isdir(Path(p)) and "constructed" not in p:
        print(p)
        dsc = datasets.load_from_disk(f"{p}/correct")
        dsi = datasets.load_from_disk(f"{p}/incorrect")
        if os.path.exists(Path(p+"/incorrect_ood")):
            dso = datasets.load_from_disk(f"{p}/incorrect_ood")
            programs_o = set(dso["hexsha"])
        else:
            programs_o = None
        programs_c = set(dsc["hexsha"])
        programs_i = set(dsi["hexsha"])
        
        print(len(programs_c.difference(programs_i)))
        print(len(programs_i.difference(programs_c)))
        if programs_o != None:
            assert len(programs_c.intersection(programs_o)) == 0
            assert len(programs_i.intersection(programs_o)) == 0
