from picarx import Picarx
import time
import threading
import readchar 
import os

px = Picarx()
#print( px.line_reference,"reference")
try:
  while True:
    gm_val_list = px.get_grayscale_data()
    print("gm_val_list:",gm_val_list)
    gm_status = px.get_line_status(gm_val_list)
    print("gm_status:",gm_status)
except:
  px.stop()
