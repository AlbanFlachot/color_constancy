#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Mar  3 14:17:28 2020

@author: alban
"""

import subprocess
import shlex

gpu = 0

load_dir = '/mnt/juggernaut/alban/project_color_constancy/PYTORCH/WCS/train_centered/All_muns/'

save_dir = '/home/alban/works/color_constancy/All_muns/scripts_server/'

train_sets = ['CC','D65']

test_conditions = ['normal','no_patch', 'wrong_illu', 'no_back']

layers = ['fc2','fc1', 'c3', 'c2', 'c1']

'''
for train_condition in train_sets:
	for test_condition in test_conditions:
	    command = "python Activations_readouts.py --gpu_id %i --model Original --testing_set WCS --testing_type 4illu --testing_condition %s --training_set %s --load_dir %s" %(gpu, test_condition, train_condition, load_dir)
	    print(command)
	    args = shlex.split(command)
	    subprocess.call(args)'''
'''	    
command = "python -i Activations_readouts.py --gpu_id %i --model Original --testing_set WCS --testing_type D65 --testing_condition normal --training_set %s --load_dir %s" %(gpu, train_sets[1], load_dir)
print(command)
args = shlex.split(command)
subprocess.call(args)'''

### Testing script

for test_condition in test_conditions:
	command = "python -i Testing_script.py --gpu_id %i --model Original --testing_set WCS --testing_type 5illu --testing_condition normal --training_set CC --load_dir %s --save_dir %s" %(gpu, load_dir, save_dir)
	print(command)
	args = shlex.split(command)
	subprocess.call(args)

