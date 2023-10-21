from picarx import Picarx
import time
import threading
import readchar 
import os

px = Picarx()
while true:
  temp=px.get_grayscale_data()
  print(temp)
