# core libraries
import os
import sys

# dependencies

# add /src and /lib to sys.path
sys.path.insert(1, os.path.join(os.getcwd(), 'src'))
sys.path.insert(1, os.path.join(os.getcwd(), 'physical-modelling-lib/build'))

# import project files
import physical_lib

print(physical_lib.add(2, 3))
