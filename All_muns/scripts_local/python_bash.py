#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Mar  3 14:17:28 2020

@author: alban
"""


import subprocess
import shlex

net = ['Original', 'Ref']

training_set = ['CC', 'D65']

conditions = ['normal','no_patch', 'wrong_illu', 'no_back', 'D65']

layers = ['fc2','fc1', 'c3', 'c2', 'c1']

for layer in layers:
    for condition in conditions[:2]:
    	command = "python compute_error.py --NetType Original --training_set CC --testing_set WCS --testing_type 4illu --testing_condition %s --path2activations /home/alban/mnt/DATA/project_color_constancy/data/training_centered/All_muns/outs/out --layer %s" %(condition, layer)
    	print(command)
    	args = shlex.split(command)
    	subprocess.call(args)


#command = "python filters.py --training_set %s --NetType Original " %('D65')
#print(command)
#args = shlex.split(command)
#subprocess.call(args)
