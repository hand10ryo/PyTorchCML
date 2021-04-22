import os
import sys

pathdir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
pathdir = os.path.join(pathdir, "src")
sys.path.append(pathdir)
