import os
import sys
sys.path.append("H:/projects-repo/024_bif_care/bif_care/")

import glob
from bif_care import gui
from bif_care.core import BifCareTrainer


project_fn = "H:/projects-repo/024_bif_care/bif_care/ste_out_rob/bif_care.json"

predict_dir = "J:/heisegrp/SteScratch/To BIF/to_Robert/Time lapse to be done"

gui.params.load(project_fn)

bt = BifCareTrainer(**gui.params)
for pf in glob.glob(predict_dir + "/*pt[3,4].czi"):
    print(pf)
    print("*"*30)
    bt.predict(pf, n_tiles=(1,4,4))


print("done")