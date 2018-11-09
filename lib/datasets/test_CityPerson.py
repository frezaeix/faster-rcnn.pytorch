import sys
import os
from IPython import  embed
lib_dir = os.path.abspath(os.path.join(os.path.dirname( __file__ ), os.pardir))
sys.path.insert(0, lib_dir)

from datasets.city_person import city_persnon

d = city_persnon('train')
roidb = d.roidb