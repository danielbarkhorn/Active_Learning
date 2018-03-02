import os
import sys

src_dir = os.path.join(os.getcwd(), os.pardir)
sys.path.append(src_dir)

from data.dataset import Dataset

Dataset('SUSY_100k.csv')
