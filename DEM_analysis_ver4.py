#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 10 08:58:59 2021

Questo script analizza tutti idem all'interno della cartella,
ne valuta l'andamento delle quote lungo l'asse longitudinale e ne
calcola la pendenza residua.
Per l'applicazione del modell di Engelund il calcolo della sezione equivalente
cviene effettuato su un DEM detrended, a cui è stata tolta la pendenza residua.
Plotta la adistribuzione di frequenza dei valori di quota.
Realizza la sezione equivalente da tutti i punti di ogni DTM e
implementa il metodo di Engelund che, raggiunta la portata di riferimento,
fornisce la Wactive e la Wwet.

@author: erri
"""
import os
import numpy as np
import matplotlib.pyplot as plt
from pyEngelund_chezy import *

# Script parameters:
nan = -999


# setup working directory and DEM's name
w_dir = os.getcwd()
input_dir = os.path.join(w_dir, 'surveys')
files=[]
for f in sorted(os.listdir(input_dir)):
    path = os.path.join(input_dir, f)
    if os.path.isfile(path) and f.startswith('matrix_bed_norm') and f.endswith('.txt'):
        files = np.append(files, f)

# Set output plot directory
if os.path.exists(os.path.join(w_dir, 'plots')):
    pass
else:
    os.mkdir(os.path.join(w_dir, 'plots'))


# array mask for filtering data outside the channel domain
# TODO Verify shape and masking properties of the mask
array_mask, array_mask_path = 'array_mask.txt', w_dir
mask0 = np.loadtxt(os.path.join(array_mask_path, array_mask), skiprows=8)
array_msk = np.where(np.isnan(mask0), 0, 1)
array_msk_nan = np.where(np.logical_not(np.isnan(mask0)), 1, np.nan)


slope = []
# For single DEM run, set files
# files = ['matrix_bed_norm_q10S0.txt']
# files loop stars here
for f in files:
    path = os.path.join(input_dir, f)
    DEM = np.loadtxt(path,
                  # delimiter=',',
                  skiprows=8
                  )
    DEM = np.where(np.isclose(DEM, nan), np.nan, DEM)
    # Shape control:
    arr_shape=min(DEM.shape, array_msk_nan.shape)
    if not(DEM.shape == array_msk_nan.shape):
        print()
        print('Analizing DEM ', f)
        # print('Attention: DEM do not have the same shape.')
        # print('Reshaping...')
        # reshaping:
        rows = min(DEM.shape[0], array_msk_nan.shape[0])
        cols = min(DEM.shape[1], array_msk_nan.shape[1])
        arr_shape = [rows, cols]
    # Reshaping:
    DEM=DEM[0:arr_shape[0], 0:arr_shape[1]]
    array_msk_nan=array_msk_nan[0:arr_shape[0], 0:arr_shape[1]]
    # Masking:
    DEM_msk = DEM*array_msk_nan
    # print(DEM_msk.shape)

    # Set run parameters:
    vpoints = 500 # N punti discretizzazione verticale per il calcolo delle Q da inserire nella scala delle portate
    g = 9.81 # Accelerazione di gravita'
    iF = 0.01 # Pendenza
    NG = 4 # Numero di punti di Gauss
    tol = 1e-03 # Tolleranza nel calcolo della profondita' critica
    tol_Q = 1e-06 # Tolleranza nel calcolo della portata di riferimento
    MAXITER = 1000 # Numero massimo di iterazioni nel calcolo della profondità critica
    fs = 10 # Font Size x grafici
    teta_c = 0.02 # [-] Shields parameter 0.03 0.035 0.040
    ds = 0.001 # [m] Sediment diameter
    Q_ref = float(f[17:19])/10000 # reading Q value from DEM name
    print('Run discharge [m**3/s]: ', Q_ref)
    W = 0.6 # Channel width [m]
    dx = 50 #[mm] longitudinal laser survey discretization
    dy= 5 #[mm] cross section laser survey discretization
    ks = 40 # Roughness factor Gauckler-Strickler
    k = 5.3 # C = 2.5*ln(11*D/(k*ds))
    sezione = str(f[16:21])

    #Calculating residual slope:
    sect_mean=[]

    for i in range (0, DEM_msk.shape[1]):
        sect_mean=np.append(sect_mean, np.nanmean(DEM_msk[:,i]))
        #print(sect_mean)

    # Performing linear regression
    x_coord = np.linspace(0, dx*len(sect_mean), len(sect_mean))
    linear_model = np.polyfit(x_coord, sect_mean,1)

    # PLOT cross section mean values and trendline
    fig, ax1 = plt.subplots(dpi=200)
    ax1.plot(x_coord, sect_mean)
    ax1.plot(x_coord, x_coord*linear_model[0]+linear_model[1], color='red')
    ax1.set(xlabel='longitudinal coordinate (mm)', ylabel='Z (mm)',
           title=str(f[16:21])+'\n'+'Residual slope:'+str(linear_model[0]))
    slope = np.append(slope, linear_model[0])

    # DEM detrending:
    # Removing the residual slope before creating equivalent cross section

    for k in range (0, DEM_msk.shape[1]):
        for j in range (0, DEM_msk.shape[0]):
            DEM_msk[j,k] = DEM_msk[j,k] -linear_model[0]*k*dx

    sect_mean_detrend=[]

    for i in range (0, DEM_msk.shape[1]):
        sect_mean_detrend=np.append(sect_mean_detrend, np.nanmean(DEM_msk[:,i]))
    linear_model_detrend = np.polyfit(x_coord, sect_mean_detrend,1)

    # PLOT cross section mean values and trendline of detrended DEM
    fig, ax1 = plt.subplots(dpi=200)
    ax1.plot(x_coord, sect_mean_detrend)
    ax1.plot(x_coord, x_coord*linear_model_detrend[0]+linear_model_detrend[1], color='green')
    ax1.set(xlabel='longitudinal coordinate (mm)', ylabel='Z (mm)',
           title=str(f[16:21])+'\n'+'Residual slope detrended:'+str(linear_model_detrend[0]))

    # Slope after detrending
    slope_detrend = np.append(slope, linear_model[0])





    # Distribution of frequency
    unique_elements, counts_elements = np.unique(DEM_msk[~np.isnan(DEM_msk)], #Avoid Nan counting
                                                 return_counts=True)
    # Sorting counts_elements keeping the match with the unique element
    # unique_elements_sort = unique_elements[np.argsort(counts_elements)]
    # counts_elements_sort = np.sort(counts_elements)

    # PLOT distribution of frequency of Z values
    fig, ax2 = plt.subplots(dpi=200)
    ax2.plot(unique_elements, counts_elements)
    ax2.set(xlabel='Z value [mm]', ylabel='f', title=str(f[16:21])+'\n'+'Frequency distribution')

    # Creating equivalent cross-section
    DTM_flat = sorted(DEM_msk[~np.isnan(DEM_msk)].flatten())# Flatten DTM values
    s=50
    DTM_flat_resize = np.array(DTM_flat[::s])
    # Build cross section banks
    DTM_flat_resize = np.pad(DTM_flat_resize, (1,1), mode = 'constant', constant_values=int(DTM_flat_resize.max()*2))
    # DTM_flat_resize[0], DTM_flat_resize[-1] = DTM_flat_resize.max()*2, DTM_flat_resize.max()*2




    # Perform mean over n_averaged_elements elements in array
    # DTM_flat_resize = []
    # n_averaged_elements = 10
    # for i in range(0, len(DTM_flat), n_averaged_elements):
    #     slice_from_index = i
    #     slice_to_index = slice_from_index + n_averaged_elements
    #     DTM_flat_resize.append(np.mean(DTM_flat[slice_from_index:slice_to_index]))

    # Perform a deleting element every n alements in array
    # n = 2
    # DTM_flat_resize = np.delete(DTM_flat, np.arange(0, DTM_flat.size, n))
    # crs_sect= np.linspace(0, W, len(DTM_flat))
    # Y_resize = np.linspace(0, W, len(DTM_flat_resize))


    fig, ax3 = plt.subplots(dpi=200)
    ax3.plot(np.linspace(0, W, len(DTM_flat_resize)), DTM_flat_resize, label='Section')
    ax3.plot(np.linspace(0, W,len(DTM_flat)), DTM_flat, label='Cumulate')
    ax3.set(xlabel='Width (m)', ylabel='Z (mm)', title=str(f[16:21])+'\n'+'Equivalent cross-section')
    ax3.legend()



#     ###############################
#     # Engelund Gauss
#     ###############################

    # Carica File di Input
    # --------------------
    y = np.linspace(0, W, len(DTM_flat_resize))
    # TODO remove
    z = DTM_flat_resize/1000
    h_min = k*ds/11 # Tirante minimo per la formua di Chezy, sotto questo valore C è nullo o negativo

    # Calcolo della scala di deflusso
    # -------------------------------
    h = np.linspace( z.min() + h_min, z.max(), vpoints+1 )[1:] # Array dei livelli della superficie libera (ma salto il primo: tirante nullo portata nulla)

    # TODO senza fare il ciclo su tutti i tiranti basterebbe implementare un metodo dicotomico partendo da un valore di primo tentativo
    Y = h - z.min() # Array dei tiranti

    # Inizializzo gli array di output
    # -------------------------------
    Q = np.zeros(vpoints) # Portata
    Omega = np.zeros( vpoints ) # Area
    b = np.zeros( vpoints ) # Larghezza superficie libera
    B = np.zeros( vpoints )
    alpha = np.zeros( vpoints ) # Coefficiente di ragguaglio dell'energia
    beta = np.zeros( vpoints ) # Coefficiente di ragguaglio della qta' di moto
    Yc = np.zeros( vpoints ) # Tirante critico
    Qs = np.zeros( vpoints ) # Portata solida
    count_active = np.zeros ( vpoints )

    # Ciclo sui tiranti
    # -----------------
    # Per ciascun livello della superficie libera calcolo la portata defluente in condizioni
    # di moto uniforme, i coefficienti di ragguaglio e la geometria della sezione bagnata


    for n in range( vpoints ):
        # Calcola i parametri di moto uniforme assegnato il tirante
        Q[n], Omega[n], b[n], B[n], alpha[n], beta[n], Qs[n], count_active[n] = MotoUniforme( iF, y, z, Y[n], NG, teta_c, ds )

        #print('tirante:', Y[n], '    Portata:', Q[n])
        # Calcola il livello critico associato alla portata corrispondente
        # al livello di moto uniform corrente
        #Yc[n] = Critica( iF, y, z, Q[n], NG, MAXITER, tol, teta_c, ds)

        #Calcolo del tirante critico
        #Yc_ana = (Q[n]**2/(g*b[n]**2))**(1/3)
# TODO implementare metodo bisezione
        if abs(Q_ref-Q[n]) < tol_Q or Q[n]>Q_ref:
            print('Q_ref reached: ', Q[n])
            break

    # Resize arrays
    Y=Y[0:n]
    Q=Q[0:n]
    Qs=Qs[0:n]
    Yc=Yc[0:n]
    Omega=Omega[0:n]
    b=b[0:n]
    alpha=alpha[0:n]
    beta=beta[0:n]
    count_active=count_active[0:n]
    #    print("Tirante critico numerico = %f Tirante analitico = %f" % (Yc[n],Yc_ana))

    # Calculate W_w
    W_wet_c = np.where(z-z.min()<=Y[-1], 1, 0) # W_w array
    W_wet = np.count_nonzero(W_wet_c == 1)/len(DTM_flat_resize)
    # print('W_w = ', W_wet*W, 'm')
    # print('Wwet/W = ', W_wet)

    # Calculate W_active
    #W_active = np.where( array_teta>teta_c, 1, 0)
    W_active = count_active[-1]/len(DTM_flat_resize)
    # print('W_active = ', W_active*W, 'm')
    # print('W_Active/W = ', W_active)
    print('Wwet/w ', W_wet, 'Wactive/W ', W_active)


# TODO Complete results print to .txt
# Save results
# header = 'Y, Q, W_wet, W_active'
# out_table1 = np.array([Y, Q, W_wet, W_active ]).T
# np.savetxt( w_dir+'out.txt', out_table1, header=header )
# print(out_table1)

#Print Qs array
#print(*Qs)

# Salva File di Output
# --------------------
# out_table = np.array([ Y, Q, Yc, b, B,Omega, alpha, beta, Qs]).T
# np.savetxt( file_output, out_table )


# Crea i grafici
# --------------
fig, ax = plt.subplots(2,2)

#Sezione y vs. z
ax[0,0].plot(y, z-z.min(),'--.', color='sienna', label = 'Sezione') #Sezione trasversale
ax[0,0].set_xlabel( 'y [m]', fontsize=fs )
ax[0,0].set_ylabel( 'z [m]', fontsize=fs )
ax[0,0].grid()
ax[0,0].set_title('Sezione ' + sezione, fontsize=1.2*fs)
ax[0,0].legend(loc='best', fontsize=0.5*fs)

#Scala di deflusso. Q vs Y, Yc
#ax[0,1].plot(Qs, Y, color='mediumblue', label = 'Tirante') #Scala delle portate solide
ax[0,1].plot(Q, Y, color='mediumblue', label = 'Tirante') #Scala delle portate
#ax[0,1].plot(Q, Yc, color='darkred', label = 'Tirante critico') #Scala delle portate-Tirante critico
ax[0,1].set_xlabel( 'Q [m^3/s]', fontsize=fs )
ax[0,1].set_ylabel( 'Y [m]', fontsize=fs )
ax[0,1].grid()
ax[0,1].set_title('Scala portate', fontsize=1.2*fs)
ax[0,1].legend(loc='best', fontsize=0.5*fs)

#Omega, b vs. Y
ax[1,0].plot(Omega, Y, color='gold', label = 'Omega')
ax[1,0].plot(b, Y, color='rebeccapurple', label = 'Larghezza pelo libero')
ax[1,0].set_xlabel( 'Omega [m^2], b [m]', fontsize=fs )
ax[1,0].set_ylabel( 'Y [m]', fontsize=fs )
ax[1,0].grid()
ax[1,0].set_title('Omega, b vs. Y', fontsize=1.2*fs)
ax[1,0].legend(loc='best', fontsize=0.5*fs)

#alpha, beta vs. Y
ax[1,1].plot(alpha, Y, color='dodgerblue', label = 'alpha')
ax[1,1].plot(beta, Y, color='darkgreen', label = 'beta')
ax[1,1].set_xlabel( 'alpha, beta', fontsize=fs )
ax[1,1].set_ylabel( 'Y [m]', fontsize=fs )
ax[1,1].grid()
ax[1,1].set_title('alpha, beta vs. Y', fontsize=1.2*fs)
ax[1,1].legend(loc='best', fontsize=0.5*fs)

plt.tight_layout() # Ottimizza gli spazi
plt.subplots_adjust(top=0.9)
plt.savefig(w_dir + '/plots/plot_' + sezione + '.pdf', dpi=1000)
plt.show()

#print slopes array
print(slope)





# #Plot Sezione
# #Scala di deflusso. Q vs Y, Yc
# fig, ax = plt.subplots(figsize=(12, 12))
# #ax.plot(Q, Y, color='mediumblue', label = 'Tirante') #Scala delle portate

# #ax.set_xlim([0, 20])
# ax.plot(y, z-z.min(),'--.', color='sienna', label = 'Sezione') #Sezione trasversale
# ax.set_xlabel( 'y [m]', fontsize=2*fs )
# ax.set_ylabel( 'z [m]', fontsize=2*fs )
# ax.grid()
# ax.set_title('Sezione ' + sezione, fontsize=3*fs)
# ax.legend(loc='best', fontsize=2*fs)

# plt.grid()
# ax.set_title(sezione, fontsize=3*fs)
# ax.legend(loc='best', fontsize=fs)
# plt.tight_layout() # Ottimizza gli spazi
# plt.subplots_adjust(top=0.9)
# plt.savefig(w_dir + '/plot/plot_' + sezione + "2" + '.png', dpi=200)
# plt.show()

# #Plot Qs
# #Scala di deflusso. Q vs Y, Yc
# fig, ax = plt.subplots(figsize=(12, 12))
# ax.plot(Q, Y, color='mediumblue', label = 'Portata') #Scala delle portate
# #ax.plot(Qs, Y, color='darkred', label = 'Trasporto Solido') #Scala delle portate-Tirante critico
# ax.set_xlabel( 'Q [m^3/s]', fontsize=2*fs )
# ax.set_ylabel( 'Y [m]', fontsize=2*fs )
# #ax.set_xlim([0, 20])
# plt.grid()
# ax.set_title('Scala di deflusso ' + sezione, fontsize=3*fs)
# ax.legend(loc='best', fontsize=fs)
# plt.tight_layout() # Ottimizza gli spazi
# plt.subplots_adjust(top=0.9)
# plt.savefig(w_dir + '/plot/plot_' + sezione + "3" + '.png', dpi=200)
# plt.show()

# #Plot Qs
# #Scala di deflusso. Q vs Y, Yc
# fig, ax = plt.subplots(figsize=(12, 12))
# #ax.plot(Q, Y, color='mediumblue', label = 'Tirante') #Scala delle portate
# ax.plot(Qs, Y, color='darkred', label = 'Trasporto Solido') #Scala delle portate-Tirante critico
# ax.set_xlabel( 'Qs [m^3/s]', fontsize=2*fs )
# ax.set_ylabel( 'Y [m]', fontsize=2*fs )
# #ax.set_xlim([0, 20])
# plt.grid()
# ax.set_title('Scala trasporto solido ' + sezione, fontsize=3*fs)
# ax.legend(loc='best', fontsize=fs)
# plt.tight_layout() # Ottimizza gli spazi
# plt.subplots_adjust(top=0.9)
# plt.savefig(w_dir + '/plot/plot_' + sezione + "1" + '.png', dpi=1000)
# plt.show()
