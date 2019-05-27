import os
import sys
sys.path.append("H:/projects-repo/024_bif_care/bif_care/")

import glob
from bif_care import gui
from bif_care.core import BifCareTrainer


project_fn = "H:/projects-repo/024_bif_care/bif_care/ste_out_rob/bif_care.json"

gui.params.load(project_fn)

bt = BifCareTrainer(**gui.params)

bt.predict("J:/heisegrp/SteScratch/To BIF/to_Robert/20x timelapse GFP + H2A-mCheery/!To be done/190312/190312_SeboxGFP_H2AmCherry-03.czi", n_tiles=(1,4,4))
bt.predict("J:/heisegrp/SteScratch/To BIF/to_Robert/20x timelapse GFP + H2A-mCheery/!To be done/190313/190313_SeboxGFP_H2AmCherry_Sox17Lyn-tdT-01.czi", n_tiles=(1,4,4))

print("done")