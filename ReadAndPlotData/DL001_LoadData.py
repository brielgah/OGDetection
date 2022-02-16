#!/usr/bin/env python2

# -*- coding: utf-8 -*-
"""
Created on Thu Nov  8 20:13:55 2018

@author: gravwaves
"""





# *********************************************
# IMPORT LIBRARIES
# *********************************************
import CWB_ReadLogJobs, CWB_ReadStrain
import numpy as np
import sys
import pylab
import matplotlib.pyplot as plt
from pycbc import types
from gwpy.timeseries import TimeSeries
from pathlib import Path
import pickle

# *********************************************
# INITIALIZE PARAMETERS
# *********************************************
#
# -------------------------------
# Parameters
#
FolderName        = 'O2_L1H1_RUN2_SIM_SCH'   # (O2_L1H1_RUN2_SIM_SCH,O2_L1H1_RUN3_SIM_SCH)
pc                = 'ligocluster'                    # ('mac','ligocluster')

# -----------------------------
# Data Path
#
if   pc == 'jacs':
    DataPath          = '/home/gabriel/Documentos/DELFIN/Jobs/'
elif pc == 'ligocluster':
    DataPath          = '/home/gabriel/Documentos/DELFIN/Jobs/'
# -----------------------------
# Sampling frequency
#
if   FolderName[8:11] == 'RUN' or FolderName[8:11] == 'SEG' or FolderName[8:14] == 'HYBRID':
    fs                = 2048  
                     
else:
    print('PILAS PERRITO: define sampling frequency for the selected project')
    sys.exit()     

# -----------------------------
# Lower frequency of the detector
#
flow              = 10                            

# -------------------------------
# Debugging parameters
#
doplot            = 1
doprint           = 1





# *********************************************
# DEFINE JOBS AND FACTORS
# *********************************************
#
# -------------------------------
# JOBS & FACTORS
#
# Define jobs for the selected FolderName 
if   FolderName == 'O2_L1H1_RUN2_SIM_SCH':
    JOBS   = np.array([2,3,4,5])
elif FolderName == 'O2_L1H1_RUN3_SIM_SCH':
    JOBS   = np.array([2,3,4,5,6,7,8,9,10,11,12,13,14,15])
else:
    print('PILAS PERRITO: Unknown FolderName')
    sys.exit()
    
# Define factors
if   FolderName[8:11] == 'RUN' or FolderName[8:11] == 'SEG' or FolderName[8:14] == 'HYBRID':
    FACTORS = np.array(['0.03','0.04','0.06','0.07','0.10','0.13','0.18','0.24','0.32','0.42','0.56','0.75','1.00','1.33','1.78','2.37','3.16','4.22','5.62','7.50','10.00','13.34','17.78','23.71','31.62','42.17','56.23','74.99','100.00','133.35','177.83','237.14','316.23'])
    segEdge = 10;    
else:
    sys.exit()

# For debugging: choose only one job and one factor
#JOBS    = np.array([2,5])
#JOBS = np.array([2])
FACTORS = np.array(['10.00'])





# *********************************************
# NOTE
# *********************************************
# Para este script no hay necesidad de los bucles "for" que siguen a 
# continuacion, no obstante, si son importantes para las actividades siguientes





# *********************************************
# FOR EACH DATA JOB
# *********************************************
# 
strainDatosEntrenamiento = []
strainDatosPrueba = []
datasetEntrenamiento  = Path("./DatasetEntrenamiento.bin")
datasetPrueba = Path("./DatasetPrueba.bin")
if(True):
    for i_job in np.arange(0,len(JOBS)):
        print('                                               ')
        print('***********************************************')
        print('Job:         ' + str(i_job+1) + ' of ' + str(JOBS.size) + '  -  ' + str(JOBS[i_job]))
        
        # -------------------------------
        # GET THE NUMBER OF THE JOB
        job          = JOBS[i_job]
        
        print(DataPath,FolderName,job)
        # -------------------------------
        # GET GPSINI
        GPSini = CWB_ReadLogJobs.GetGPSini(DataPath,FolderName,job)
        
        
        # -------------------------------
        # GET INJECTION TIMES FOR THE CURRENT JOB (ACROSS-ALL-TEMPLATES)
        #
        # Get injection times and the name of the injected waveform
        GPSiniEdge, TinjAll, WFnameAll, TinjL1All, TinjH1All = CWB_ReadLogJobs.GetInjectionTimes(DataPath,FolderName,job,GPSini)
        
        #Check segEdge
        if segEdge != GPSini-GPSiniEdge:
            print('PILAS PERRITO: problem with segEdge')
            sys.exit() 
        
        # -------------------------------
        # GET INJECTION TIMES RE-REFERENCED TO T=0 
        Tinj                = TinjAll    - GPSiniEdge
        TinjL1              = TinjL1All  - GPSiniEdge
        TinjH1              = TinjH1All  - GPSiniEdge
        GPSiniEdge          = 0 
        
        print('# of inj:    ' + str(Tinj.size) )

        #print(str(Tinj))
        
        print('Times where injections of L1 ocurred' + str(TinjL1[:61]))
        print('Times where injections of H1 ocurred' + str(TinjH1[:61]))
        
        
        # *********************************************
        # FOR EACH FACTOR
        # *********************************************
        #
        strainDatosFactores = []
        for i_fac in np.arange(0,FACTORS.size): 
            print('Factor:      ' + str(i_fac+1) + ' of ' + str(FACTORS.size) + '  -  ' + FACTORS[i_fac] )
        
            
            # -------------------------------
            # GET CURRENT FACTOR AND COMPUTE DISTANCE
            #
            # Get the current factor
            fac               = FACTORS[i_fac]
            
            # Get current distance
            Rinj              = 10./float(fac)
                    
            print('Distance:    ' + str(i_fac+1) + ' of ' + str(FACTORS.size) + '  -  ' + str(Rinj) + 'Kpc')
            
            strainFactor = [],[],[],[],[]
            # -------------------------------
            # LOAD AND PLOT STRAIN DATA:
            strainH1raw, strainL1raw    = CWB_ReadStrain.ReadStrain(DataPath,FolderName,job,fac,GPSini,fs,'r',0)
            plt.figure(figsize=(12,6))
            plt.plot(strainH1raw.sample_times,strainH1raw,linewidth=2.0,label='H1')
            #plt.plot(strainL1raw.sample_times,strainL1raw,linewidth=2.0,label='L1')
            plt.legend()
            plt.title('Strain around injection (RAW)',fontsize=18)
            plt.xlabel('Time (s)',fontsize=18,color='black')
            plt.ylabel('Strain',fontsize=18,color='black')
            plt.grid(True)
            plt.savefig("Onda")
            plt.close()
            strainH1raw = TimeSeries.from_pycbc(strainH1raw,copy=True)
            strainL1raw = TimeSeries.from_pycbc(strainL1raw,copy=True)
            #plt.axvline(x=t_inj, color='r', linestyle='--', linewidth=1)
            #TimeSeries.bandpass(serie,flow,fhigh,gpass,gstop,fstop,type)
            #flow = float	frecuencia baja
            #fhigh = float	frecuencia alta
            #gpass = float	maxima perdida en el pasa bandas (db)
            #gstop = float	la atenuacion minima en el rechazo de banda
            #fstop = tuple o float (low,high) opcional (baja,alta) limite de 
            #freciencias en el filtro de rechazo de banda
            #type = string 'iir' o 'fir'
            # respuesta infinita al pulso o respuesta finita al pulso
            strainH1raw = TimeSeries.bandpass(strainH1raw,flow=20,fhigh=520)
            strainL1raw = TimeSeries.bandpass(strainL1raw,flow=20,fhigh=520)
            strainL1raw = TimeSeries.to_pycbc(strainL1raw)
            strainH1raw = TimeSeries.to_pycbc(strainH1raw)
            # TimeSeries.whiten(serie,segment_duration,max_filter_duration)
            #segment_duration = float Duracion en segundos de cada muestra del
            # espectro
            #max_filter_duration = float Duracion del filtro que se aplica
            #Se resta del total de duracion
            strainH1raw = types.TimeSeries.whiten(strainH1raw,segment_duration=4,max_filter_duration=4)
            strainL1raw = types.TimeSeries.whiten(strainL1raw,segment_duration=4,max_filter_duration=4)
            plt.figure(figsize=(12,6))
            plt.plot(strainH1raw.sample_times,strainH1raw,linewidth=2.0,label='H1')
            #plt.plot(strainL1raw.sample_times,strainL1raw,linewidth=2.0,label='L1')
            plt.legend()
            plt.title('Strain around injection (RAW)',fontsize=18)
            plt.xlabel('Time (s)',fontsize=18,color='black')
            plt.ylabel('Strain',fontsize=18,color='black')
            plt.grid(True)
            plt.savefig("Onda2")
            plt.close()
            strainH1raw = TimeSeries.from_pycbc(strainH1raw,copy=True)
            strainL1raw = TimeSeries.from_pycbc(strainL1raw,copy=True)
            if doprint == 1:
                print('----')
                print('Raw data duration H1: ' + str(strainH1raw.duration) + ' seconds')
                print('Raw   data duration L1: ' + str(strainL1raw.duration) + ' seconds')
                #sys.exit()
            
            # BREAK
            #sys.exit()
            
            
            # -------------------------------
            # PLOT STRAIN DATA AROUND ONE OF THE INJECTIONS:
            if doplot == 0:
                
                # Choose one of the injections
                strainData = [],[]
                strainClase = []
                strainName = []
                strainDist = []
                y = 0
                # Se crean las ventanas
                for iTemp in np.arange(0,61):
                #iTemp    =  0   # Cuidado: este numero no puede ser mayor a Tinj.size
            
            # Imprimir nombre de la injected GW
                    print('Injected GW:    ' + WFnameAll[iTemp] )
                    print("Time Injected GW: "+str(Tinj[iTemp]))
                    # Selecccionar un intervalos de tiempo alrededor del tiempo Tinj
                    # Se genera imagenes con ruido#
                    for x in range(3,0,-1):
                        t_inj    = Tinj[iTemp] - (x+2)
                        li       = t_inj - 0.25
                        lf       = t_inj + 0.25
                        
                        # Plot strain data
                        # TimeSeries.qtransform(delta_t,delta_f,longsteps,frange,qrange,mismatch,return_complex)
                        # delta_t = float El espaciado de tiempo para la imagen de salida 
                        # delta_f = float La fracuencia para la imagen de salida
                        # longsteps = int Hace una interpolacion logaritmica y establece el numero de saltos a hacer
                        # Incompatible con delta_t
                        # frange = tuple ints El rango de frecuencias
                        # qrange = tuple int rango q
                        # mismatch = float desajuste entre las baldosas de frecuencia
                        # return_complex = regresa la serie compleja cruda en lugar de la normalizada
                        # Regresa
                        # times,freqs,qplane
                        # times = numpy.ndarray El tiempo a la que la transformada q es muestreada 
                        # freqs = numpy.ndarray La frecuencia a la que la transformada q es muestreada
                        # qplane = numpy.ndarray(2d) La transformación q bidimensional interpolada de esta serie temporal
                        nuevaSerieH1 = types.TimeSeries.time_slice(strainH1raw,start=li,end=lf,mode="floor")
                        nuevaSerieL1 = types.TimeSeries.time_slice(strainL1raw,start=li,end=lf,mode="floor")
                        tiempoH1,frecuenciasH1,transQH1 = nuevaSerieH1.qtransform(frange=(20,512),qrange=(8,8))
                        tiempoRedH1,frecuenciasRedH1,transRedH1 = nuevaSerieH1.qtransform(delta_t=0.01,frange=(16,512),qrange=(8,8))
                        tiempoL1,frecuenciasL1,transQL1 = nuevaSerieL1.qtransform(frange=(20,512),qrange=(8,8))
                        tiempoRedL1,frecuenciasRedL1,transRedL1 = nuevaSerieL1.qtransform(delta_t=0.01,frange=(16,512),qrange=(8,8))
                        '''fig,axs = plt.subplots(2,figsize=[12,8])
                        axs[0].pcolormesh(tiempoRedL1,frecuenciasRedL1,transRedL1**0.5,shading='auto')
                        axs[0].set_xlim(li,lf)
                        axs[0].set_title("Noise L1")
                        axs[0].axvline(x=t_inj,color='w',linestyle='--', linewidth=1)
                        axs[1].pcolormesh(tiempoRedH1,frecuenciasRedH1,transRedH1**0.5,shading='auto')
                        axs[1].set_xlim(li,lf)
                        axs[1].set_title("Noise H1")
                        axs[1].axvline(x=t_inj,color='w',linestyle='--', linewidth=1)
                        nameFigure = str(y)+"Job"+str(i_job)+"Clase"+str(0)+"Fact%.2f"+FACTORS[i_fac]+WFnameAll[iTemp]+"Mesh.png"
                        plt.savefig(nameFigure)'''
                        #plt.show()
                        plt.close()
                        strainData[0].append(transRedH1)
                        strainData[1].append(transRedL1)
                        strainClase.append(0)
                        strainName.append(WFnameAll[iTemp])
                        strainDist.append(Tinj[iTemp])
                        '''plt.figure(figsize=(12,6))
                        plt.plot(nuevaSerieH1.sample_times,nuevaSerieH1,linewidth=2.0,label='H1')
                        plt.plot(nuevaSerieL1.sample_times,nuevaSerieL1,linewidth=2.0,label='L1')
                        plt.title('Noise',fontsize=18)
                        plt.xlabel('Time (s)',fontsize=18,color='black')
                        plt.ylabel('Strain',fontsize=18,color='black')
                        plt.grid(True)
                        plt.axvline(x=t_inj, color='r', linestyle='--', linewidth=1)
                        plt.legend()
                        nameFigure = str(y)+"Job"+str(i_job)+"Clase"+str(0)+"Fact%.2f"+FACTORS[i_fac]+WFnameAll[iTemp]+".png"        
                        plt.savefig(nameFigure)
                        plt.close()'''
                        # Plot Tinj, TinjH1, TinjL1
                        #for ti in Tinj:
                        #pylab.axvline(x=t_inj, color='r', linestyle='--', linewidth=1)
            #            for ti in TinjL1:
            #                pylab.axvline(x=ti, color='g', linestyle=':', linewidth=1)           
            #            for ti in TinjH1:
            #                pylab.axvline(x=ti, color='b', linestyle=':', linewidth=1)
                        
                        # Ajustar xlim al intervalo de interes
                        y+= 1
                        ####################################
                        # Termina generacion de ruido #
                        
                        # Se genera imagenes con ruido + OG #
                    injectionWindowLeft = [0.75,0.5,0.25]
                    injectionWindowRight = [0.25,0.5,0.75]
                    for x in range(0,3,1):
                        t_inj    = Tinj[iTemp]
                        li       = t_inj - 0.5*injectionWindowLeft[x]
                        lf       = t_inj + 0.5*injectionWindowRight[x]
                        nuevaSerieH1 = types.TimeSeries.time_slice(strainH1raw,start=li,end=lf,mode="floor")
                        nuevaSerieL1 = types.TimeSeries.time_slice(strainL1raw,start=li,end=lf,mode="floor")
                        tiempoH1,frecuenciasH1,transQH1 = nuevaSerieH1.qtransform(frange=(20,512),qrange=(8,8))
                        tiempoRedH1,frecuenciasRedH1,transRedH1 = nuevaSerieH1.qtransform(delta_t=0.01,frange=(16,512),qrange=(8,8))
                        tiempoL1,frecuenciasL1,transQL1 = nuevaSerieL1.qtransform(frange=(20,512),qrange=(8,8))
                        tiempoRedL1,frecuenciasRedL1,transRedL1 = nuevaSerieL1.qtransform(delta_t=0.01,frange=(16,512),qrange=(8,8))
                        '''fig,axs = plt.subplots(2,figsize=[12,8])
                        axs[0].pcolormesh(tiempoRedL1,frecuenciasRedL1,transRedL1**0.5,shading='auto')
                        axs[0].set_xlim(li,lf)
                        axs[0].set_title("Strain around injection H1")
                        axs[0].axvline(x=t_inj,color='w',linestyle='--', linewidth=1)
                        axs[1].pcolormesh(tiempoRedH1,frecuenciasRedH1,transRedH1**0.5,shading='auto')
                        axs[1].set_xlim(li,lf)
                        axs[1].set_title("Strain around injection L1")
                        axs[1].axvline(x=t_inj,color='w',linestyle='--', linewidth=1)
                        nameFigure = str(y)+"Job"+str(i_job)+"Clase"+str(1)+"Fact%.2f"+FACTORS[i_fac]+WFnameAll[iTemp]+"Mesh.png"        
                        plt.savefig(nameFigure)
                        #plt.show()
                        plt.close()'''
                        strainData[0].append(transRedH1)
                        strainData[1].append(transRedL1)
                        strainClase.append(1)
                        strainName.append(WFnameAll[iTemp])
                        strainDist.append(Tinj[iTemp])
                        # Plot strain data
                        #'''plt.figure(figsize=(12,6))
                        #plt.plot(nuevaSerieH1.sample_times,nuevaSerieH1,linewidth=2.0,label='H1')
                        #plt.plot(nuevaSerieL1.sample_times,nuevaSerieL1,linewidth=2.0,label='L1')
                        #plt.legend()
                        #plt.title('Strain around injection (RAW)',fontsize=18)
                        #plt.xlabel('Time (s)',fontsize=18,color='black')
                        #plt.ylabel('Strain',fontsize=18,color='black')
                        #plt.grid(True)
                        #plt.axvline(x=t_inj, color='r', linestyle='--', linewidth=1)
                        #nameFigure = str(y)+"Job"+str(i_job)+"Clase"+str(1)+"Fact%.2f"+FACTORS[i_fac]+WFnameAll[iTemp]+".png"
                        #plt.savefig(nameFigure)
                        #plt.close()'''
                        # Plot Tinj, TinjH1, TinjL1
                        #for ti in Tinj:
                            #pylab.axvline(x=ti, color='r', linestyle='--', linewidth=1)
                        #for ti in TinjL1:
                        #   pylab.axvline(x=ti, color='g', linestyle=':', linewidth=1)           
                        #for ti in TinjH1:
                        #   pylab.axvline(x=ti, color='b', linestyle=':', linewidth=1)                    
                        y+=1
                        # Termina generacion de ruido + OG
                # Terminan las distancias
            strainFactor[0].append(strainData)
            strainFactor[1].append(strainClase)
            strainFactor[2].append(strainName)
            strainFactor[3].append(strainDist)
            strainFactor[4].append(fac)
            strainDatosFactores.append(strainFactor)
        #strainDatos
        #strainDatos[0][] H1
        #strainDatos[1][] L1
        #strainClase[] Clase
        #strainName[] Nombre
        #strainFactor[0][0] strainData
        #strainFactor[0][0][1][] L1
        #strainFactor[0][0][0][] H1
        #strainFactor[1][0][] Clase
        #strainFactor[2][0][] Name
        #strainFactor[3][0][] Distancia
        #strainFactor[4][0][0] Factor
        #strainDatosFactores
        #strainDatosFactores[] Factor
        #strainDatosFactores[][0][0] Arreglo de transformaciones q de las ventanas(strainData)
        #strainDatosFactores[][0][0][0][] Transformaciones q para H1 366
        #strainDatosFactores[][0][0][1][] Transformaciones q para L1 366
        #strainDatosFactores[][1][0][] Clase de la transformacion q
        #strainDatosFactores[][2][0][] Fuente a la que pertenecen las transformaciones
        #strainDatosFactores[][3][0][] Distancia de la onda inyectada
        #strainDatosFactores[][4][0][0] Factor
        if(i_job != 3):
            strainDatosEntrenamiento.append(strainDatosFactores)
        else:
            strainDatosPrueba.append(strainDatosFactores)
    with open("./DatasetEntrenamiento.bin", "wb") as datasetfile:
        pickle.dump(strainDatosEntrenamiento,datasetfile)
    with open("./DatasetPrueba.bin", "wb") as datasetfile:
        pickle.dump(strainDatosPrueba,datasetfile)
# Break system
#sys.exit()