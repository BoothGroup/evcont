#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 31 16:26:12 2023

@author: katalar
"""

import numpy as np

import os

import pickle

from itertools import product

# Bohr
test_range = np.linspace(0.8, 3.0,40)

nroots = 3

inp_f_text = \
    '&gateway\nCoord\n \
4\nHydrogen chain coordinates in Bohr\nH  0.000000  0.000000  0.000000\n\
H  xxx  0.000000  0.000000\nH  yyy  0.000000  0.000000\n\
H  zzz  0.000000  0.000000\nBasis=sto-3g\nGroup=Nosymm\n\
&SEWARD\n&RASSCF\nnactel = 4 0 0 \ninactive = 0 \nras2 = 4\nciroot =%i %i 1\n\
&ALASKA\n NAC = 1 2\n\n'%(nroots, nroots)
    
fname = 'H4.input'

###########################################################################
def read_nac_molcas(outf):
    '''
    
    '''
    # Will contain 3 lists - CI NAC, CSF NAC, and their total sum
    all_NAC = []
    
    lines = 0
    read_ci_derivative = False
    
    tmp_list = []
    with open(outf) as f:
        for line in f.readlines():
            if 'derivative coupling' in line:
                read_ci_derivative = True
                
            if read_ci_derivative:
                     
                if lines == 2 and len(line.split()) > 2:
                    # Read the values
                    tmp_list.append([float(i) for i in line.split()[1:]])
                               
                if '------' in line:
                    lines += 1
                    
                if lines == 3:
                    lines = 0
                    read_ci_derivative = False
                    all_NAC.append(tmp_list)
                    tmp_list = []
                    
    #print(all_NAC)
    return [np.array(i) for i in all_NAC]
###########################################################################
test_NACs = {}
for i, dist_i in enumerate(test_range):
    print(i)
    
    # Save the current directory
    cwd = os.getcwd()
    
    # Create a subdirectory for the geometry
    d_drc = 'd-%.4f'%dist_i
    os.mkdir(d_drc)
    os.chdir(d_drc)
    cwd_mid = os.getcwd()

    # New input file for openMOLCAS
    new_inp = inp_f_text.replace('xxx','%.6f'%dist_i).replace('yyy','%.6f'%(2*dist_i)).replace('zzz','%.6f'%(3*dist_i))
    
    nac_i = {}
    for i,j in product(range(nroots),range(nroots)):
        if i != j:
            # Create directories for different NACs
            nac_drc = 'nac-%i%i'%(i,j)
            os.mkdir(nac_drc)
            os.chdir(nac_drc)
            
            # Make input file
            nac_inp = new_inp.replace('NAC = 1 2','NAC = %i %i'%(i+1,j+1))
            
            with open(fname, 'w+') as f:
                f.write(nac_inp)
                
            # Run calculation
            os.system('$MOLCAS/pymolcas %s > out'%fname)
            
            #os.system('sleep 2')
            
            allNAC = read_nac_molcas('out')
            nac_i['%i%i'%(i,j)] = allNAC
            
            os.chdir(cwd_mid)
            
    # Add to the dataset
    test_NACs[dist_i] = nac_i
            
    # Return to original test directory
    os.chdir(cwd)
            
# Write all to file
with open('test_NACs.pkl','wb') as f:
    pickle.dump(test_NACs,f)
            
            
            
        