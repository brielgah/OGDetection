#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Nov  8 20:13:55 2018

@author: gravwaves
"""





# =============================================
# IMPORT LIBRARIES
# =============================================
import numpy as np
import pylab
import CWB_ReadLogJobs
from pycbc import types
#import sys









# =============================================
# FUNCTION TO READ CONDITIONED STRAIN DATA
# =============================================
def Original(DataPath,FolderName,job,GPSini,fs,doplot):
    
    
    # ---------------------------------------------
    # CONSTRUT FOLDER AND FILENAME
    folder      = FolderName + '/output'
    filename    = '_' + str(GPSini) + '_1200_' + FolderName + '_job' + str(job) + '.dat'    
    
    # ---------------------------------------------
    # LOAD RAW DATA
    fullpath      = DataPath + folder + '/H1o' + filename
    H1rawArray    = np.fromfile(fullpath,dtype=float)

    fullpath      = DataPath + folder + '/L1o' + filename
    L1rawArray    = np.fromfile(fullpath,dtype=float)
    
    
    # ---------------------------------------------
    # CREATE TIME SERIES OBJECT (inicio en t=0)
    H1raw        = types.TimeSeries(initial_array=H1rawArray,delta_t=1.0/fs,epoch=0)
    L1raw        = types.TimeSeries(initial_array=L1rawArray,delta_t=1.0/fs,epoch=0)  
    
    
    # ---------------------------------------------
    # DO PLOT
    if doplot == 1:
        
        # Plot strain data a injection times        
        pylab.figure()
        pylab.plot(H1raw.sample_times,H1raw,'r',linewidth=2.0)
        pylab.plot(L1raw.sample_times,L1raw,'b',linewidth=2.0)
        pylab.xlabel('Time (s)',fontsize=18,color='black')
        pylab.ylabel('Strain',fontsize=18,color='black')
        pylab.grid(True)
        #pylab.xlim(12.7,12.75)
        pylab.show()

    # ---------------------------------------------
    # RETURN OUTPUT DATA
    return H1raw, L1raw
    
    



# =============================================
# FUNCTION TO READ RAW STRAIN DATA
# =============================================
def Raw(DataPath,FolderName,job,fac,GPSini,fs,doplot):
    
    
    # ---------------------------------------------
    # CONSTRUT FOLDER AND FILENAME
    folder      = FolderName + '/output'
    if fac == 'NO':
        filename    = '_' + str(GPSini) + '_1200_' + FolderName + '_job' + str(job) + '.dat'
    else:
        filename    = '_' + str(GPSini) + '_1200_' + FolderName + '_job' + str(job) + '_factor' + fac + '.dat'    
    
    # ---------------------------------------------
    # LOAD RAW DATA
    fullpath      = DataPath + folder + '/H1r' + filename
    H1rawArray    = np.fromfile(fullpath,dtype=float)

    fullpath      = DataPath + folder + '/L1r' + filename
    L1rawArray    = np.fromfile(fullpath,dtype=float)
    
    
    # ---------------------------------------------
    # CREATE TIME SERIES OBJECT (inicio en t=0)
    H1raw        = types.TimeSeries(initial_array=H1rawArray,delta_t=1.0/fs,epoch=0)
    L1raw        = types.TimeSeries(initial_array=L1rawArray,delta_t=1.0/fs,epoch=0)  
    
    
    # ---------------------------------------------
    # DO PLOT
    if doplot == 1:
    
        # Plot strain data a injection times        
        pylab.figure()
        pylab.plot(H1raw.sample_times,H1raw,'r',linewidth=2.0)
        pylab.plot(L1raw.sample_times,L1raw,'b',linewidth=2.0)
        pylab.xlabel('Time (s)',fontsize=18,color='black')
        pylab.ylabel('Strain',fontsize=18,color='black')
        pylab.grid(True)
        
        #pylab.xlim(12.7,12.75)
        
        # Get and plot Injection times
        if fac != 'NO':
            # Get injection times
            TimeOnset, TimeInjecAll, WavefromNames, TimeInjecL1All, TimeInjecH1All = CWB_ReadLogJobs.GetInjectionTimes(DataPath,FolderName,job,GPSini)
            
            # Referenciar a t=0
            TimeInjecAll    = TimeInjecAll   - TimeOnset;
            TimeInjecH1All  = TimeInjecH1All - TimeOnset;
            TimeInjecL1All  = TimeInjecL1All - TimeOnset;
            
            # Plot injection times
            for ti in TimeInjecAll:
                pylab.axvline(x=ti, color='r', linestyle='--', linewidth=1)
            for ti in TimeInjecL1All:
                pylab.axvline(x=ti, color='g', linestyle=':', linewidth=1)           
            for ti in TimeInjecH1All:
                pylab.axvline(x=ti, color='b', linestyle=':', linewidth=1)
        pylab.show()        
    
    # ---------------------------------------------
    # RETURN OUTPUT DATA
    return H1raw, L1raw





# =============================================
# FUNCTION TO READ CONDITIONED STRAIN DATA
# =============================================
def Conditioned(DataPath,FolderName,job,fac,GPSini,fs,doplot):
    
    
    # ---------------------------------------------
    # CONSTRUT FOLDER AND FILENAME
    folder      = FolderName + '/output'
    if fac == 'NO':
        filename    = '_' + str(GPSini) + '_1200_' + FolderName + '_job' + str(job) + '.dat'
    else:
        filename    = '_' + str(GPSini) + '_1200_' + FolderName + '_job' + str(job) + '_factor' + fac + '.dat'
    
    # ---------------------------------------------
    # LOAD RAW DATA
    fullpath      = DataPath + folder + '/H1c' + filename
    H1rawArray    = np.fromfile(fullpath,dtype=float)

    fullpath      = DataPath + folder + '/L1c' + filename
    L1rawArray    = np.fromfile(fullpath,dtype=float)
    
    
    # ---------------------------------------------
    # CREATE TIME SERIES OBJECT (inicio en t=0)
    H1raw        = types.TimeSeries(initial_array=H1rawArray, delta_t=1.0/fs, epoch=0)
    L1raw        = types.TimeSeries(initial_array=L1rawArray, delta_t=1.0/fs, epoch=0)
    
    
    # ---------------------------------------------
    # SET START TIME
    
    # Get injection times
    Tonset, Tinj, WFname, TinjL1, TinjH1 = CWB_ReadLogJobs.GetInjectionTimes(DataPath,FolderName,job,GPSini)

    # Set start time
    #H1raw.start_time = Tonset
    #L1raw.start_time = Tonset
    
    
    # ---------------------------------------------
    # DO PLOT
    if doplot == 1:
    
        # Plot strain data a injection times        
        pylab.figure()
        pylab.plot(L1raw.sample_times,L1raw,'r',linewidth=2.0)
        pylab.plot(H1raw.sample_times,H1raw,'b',linewidth=2.0)
        pylab.xlabel('Time (s)',fontsize=18,color='black')
        pylab.ylabel('Strain',fontsize=18,color='black')
        pylab.grid(True)
        
        # Get and plot Injection times
        if fac != 'NO':

            
            # Plot injection times
            for ti in Tinj:
                pylab.axvline(x=ti, color='g', linestyle='--', linewidth=1)
            for ti in TinjL1:
                pylab.axvline(x=ti, color='r', linestyle=':', linewidth=1)           
            for ti in TinjH1:
                pylab.axvline(x=ti, color='b', linestyle=':', linewidth=1)
        
                
        # Define xlims
        t_inj    = Tinj[1]
        li       = t_inj - 0.1
        lf       = t_inj + 0.1
        li       = 0
        lf       = 1220
        pylab.xlim(li,lf)
        pylab.ylim(-500,500)
        pylab.show()
        
    # ---------------------------------------------
    # RETURN OUTPUT DATA
    return H1raw, L1raw
    
    

# =============================================
# FUNCTION TO READ CONDITIONED STRAIN DATA
# =============================================
def ReadStrain(DataPath,FolderName,job,fac,GPSini,fs,Type,doplot):
    

    #    # ---------------------------------------------
    #    # VERIFY THAT TYPE==o,r,c
    #    if Type != 'o' or Type != 'r' or Type != 'c':
    #        print('PILAS: unknown type of data. Type should be either o, r, or c')
    #        print('Type is: ' + Type)
    #        sys.exit()
    
    
    # ---------------------------------------------
    # CONSTRUT FOLDER AND FILENAME
    folder      = FolderName + '/output'
    if fac == 'NO':
        filename    = '_' + str(GPSini) + '_1200_' + FolderName + '_job' + str(job) + '.dat'
    else:
        filename    = '_' + str(GPSini) + '_1200_' + FolderName + '_job' + str(job) + '_factor' + fac + '.dat'
    
    
    # ---------------------------------------------
    # LOAD DATA
    fullpath      = DataPath + folder + '/H1' + Type + filename
    H1rawArray    = np.fromfile(fullpath,dtype=float)

    fullpath      = DataPath + folder + '/L1'+ Type + filename
    L1rawArray    = np.fromfile(fullpath,dtype=float)
    
    
    # ---------------------------------------------
    # REDEFINE SAMPLING FREQUENCY IF DATA IS ORIGINAL
    if Type=='o' or Type=='t' :
        fs = 16384
    
    
    # ---------------------------------------------
    # CREATE TIME SERIES OBJECT (ONSET IN t=0)
    H1raw        = types.TimeSeries(initial_array=H1rawArray, delta_t=1.0/fs, epoch=0)
    L1raw        = types.TimeSeries(initial_array=L1rawArray, delta_t=1.0/fs, epoch=0)
    
    
    # ---------------------------------------------
    # SET START TIME (ONSET IN t=GPSini)
    if fac != 'NO':
        # Get injection times
        Tonset, Tinj, WFname, TinjL1, TinjH1 = CWB_ReadLogJobs.GetInjectionTimes(DataPath,FolderName,job,GPSini)
        Tinj                = Tinj    - Tonset
        TinjL1              = TinjL1  - Tonset
        TinjH1              = TinjH1  - Tonset
        Tonset              = 0
        #        # Set start time
        #        H1raw.start_time = Tonset
        #        L1raw.start_time = Tonset
        #    else:
        #        # Set start time
        #        H1raw.start_time = GPSini
        #        L1raw.start_time = GPSini
    
    
    # ---------------------------------------------
    # DO PLOT
    if doplot == 1:
        # Plot strain data a injection times     
        pylab.figure(figsize=(12,4))
        pylab.plot(L1raw.sample_times,L1raw,linewidth=2.0 , label='L1')
        pylab.plot(H1raw.sample_times,H1raw,linewidth=2.0 , label='H1')
        pylab.legend()
        if Type=='r':
            pylab.title('Strain (RAW)',fontsize=18)
        elif Type=='c':
            pylab.title('Strain (CLEANED)',fontsize=18)
        pylab.xlabel('Time (s)',fontsize=18,color='black')
        pylab.ylabel('Strain',fontsize=18,color='black')
        pylab.grid(True)
        
        # Get and plot Injection times
        if fac != 'NO':            
            # Plot injection times
            for ti in Tinj:
                pylab.axvline(x=ti, color='g', linestyle='--', linewidth=1)
            for ti in TinjL1:
                pylab.axvline(x=ti, color='r', linestyle=':', linewidth=1)           
            for ti in TinjH1:
                pylab.axvline(x=ti, color='b', linestyle=':', linewidth=1)
                
        pylab.show()
    # ---------------------------------------------
    # RETURN OUTPUT DATA
    return H1raw, L1raw


