#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Mar  3 14:17:28 2020

@author: alban
"""

import subprocess
import shlex
import os

gpu = 3

os.environ["CUDA_VISIBLE_DEVICES"]=str(gpu)

load_dir = '/mnt/juggernaut/alban/project_color_constancy/PYTORCH/WCS/train_centered/All_muns/'

save_dir = '/home/alban/works/color_constancy/All_muns/scripts_server/'

train_sets = ['CC','D65']

test_conditions = ['normal','no_patch', 'wrong_illu', 'no_back']

layers = ['_layer1','_layer2','_layer3']

models = ['RefResNet','MobileNet', 'ResNet11', 'ResNet18', 'ResNet50', 'VGG11_bn']

'''
for train_condition in train_sets:
	for test_condition in test_conditions:
	    command = "python Activations_readouts.py --gpu_id %i --model Original --testing_set WCS --testing_type 4illu --testing_condition %s --training_set %s --load_dir %s" %(gpu, test_condition, train_condition, load_dir)
	    print(command)
	    args = shlex.split(command)
	    subprocess.call(args)'''
        

command = "python -i Activations_readouts.py --gpu_id %i --model Original --testing_set WCS --testing_type 4illu --testing_condition wrong_illu --training_set CC --load_dir %s" %(gpu, load_dir)
print(command)
args = shlex.split(command)
subprocess.call(args)

'''
### Testing script
for layer in layers[:1]:
	for condition in test_conditions:
		    command = "python Testing_script.py --gpu_id %i --model RefResNet --testing_set WCS --testing_type 5illu --testing_condition %s --training_set CC --load_dir %s --save_dir %s --focus all --layer %s" %(gpu, condition, load_dir, save_dir, layer)
		    print(command)
		    args = shlex.split(command)
		    subprocess.call(args)
'''

'''
for condition in test_conditions:
	    command = "python Testing_script.py --gpu_id %i --model RefResNet --testing_set WCS --testing_type 5illu --testing_condition %s --training_set CC --load_dir %s --save_dir %s --focus all" %(gpu, condition, load_dir, save_dir)
	    print(command)
	    args = shlex.split(command)
	    subprocess.call(args)'''

'''
command = "python  Testing_script.py --gpu_id %i --model AlbanNet --testing_set WCS --testing_type 5illu --testing_condition normal --training_set CC --load_dir %s --save_dir %s --focus all" %(gpu, load_dir, save_dir)
print(command)
args = shlex.split(command)
subprocess.call(args)
'''