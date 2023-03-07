#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 20 13:33:42 2022

@author: erri
"""

# Libreries
import numpy as np

Q_target = 0.002
toll = 0.00001

#Set functiones variables:
g = 9.806
ds = 0.001
S = 0.001
y_coord = np.array([0,0,0.6,0.6])
z_coord = np.array([1,0,0,1])
teta_c = 0.02
NG=4
D = 0.2
max_iter = 100000


# ========
# Funzioni
# ========


def GaussPoints(NG):
    '''
    Funzione per il calcolo dei punti e dei pesi di Gauss
    
    Argomenti
    ---------
    NG: int
       numero di punti di Gauss

    Output
    ------
    p: numpy.ndarray
      array dei punti di Gauss
    w: numpy.ndarray
      array dei pesi
    '''
    p, w = None, None
    if NG==2:
        p = np.array([ -1/np.sqrt(3),
                       +1/np.sqrt(3) ])
        w = np.array([ 1, 1 ])
    elif NG==3:
        p = np.array([-(1/5)*np.sqrt(15),
                      0,
                      (1/5)*np.sqrt(15)])
        w = np.array([5/9, 8/9, 5/9])
    elif NG==4:
        p = np.array([+(1/35)*np.sqrt(525-70*np.sqrt(30)),
                      -(1/35)*np.sqrt(525-70*np.sqrt(30)),
                      +(1/35)*np.sqrt(525+70*np.sqrt(30)),
                      -(1/35)*np.sqrt(525+70*np.sqrt(30))])
        w = np.array([(1/36)*(18+np.sqrt(30)),
                      (1/36)*(18+np.sqrt(30)),
                      (1/36)*(18-np.sqrt(30)),
                      (1/36)*(18-np.sqrt(30))])

    return p, w


def MotoUniforme( S, y_coord, z_coord, D, NG, teta_c, ds):
    '''
    Calcola i parametri di moto uniforme per assegnato tirante

    Argomenti
    ---------

    S: float
       pendenza del canale
    y_coord: numpy.ndarray
      coordinate trasversali dei punti della sezione
    z_coord: numpy.ndarray
      coordinate verticali dei punti della sezione
    D: float
      profondità alla quale calcolare i parametri di moto uniforme
    NG: int [default=2]
      numero di punti di Gauss
    teta_c: float
        parametro di mobilità critico di Shiels
    ds: float
        diamentro medio dei sedimenti

    Output
    ------
    Q: float
      portata alla quale si realizza la profondità D di moto uniforme
    Omega: float
      area sezione bagnata alla profondita' D
    b: float
      larghezza superficie libera alla profondita' D
    alpha: float
      coefficiente di ragguaglio dell'energia alla profondita' D
    beta: float
      coefficiente di ragguaglio della qdm alla profondita' D
    '''
    # Punti e pesi di Gauss
    xj, wj = GaussPoints( NG ) # Calcola i putni e i pesi di Gauss
    
    #Dati
    delta = 1.65
    g = 9.806
    k = 5.3 # C = 2.5*ln(11*D/(k*ds))

    # Inizializzo
    Omega = 0 # Area bagnata
    array_teta = [] # Shields parameter array
    b = 0 # Larghezza superficie libera
    sumQs = 0 # Portata solida
    B=0
    #I coefficienti di ragguaglio sono relativi a tutta la sezione, si calcolano alla fine.
    num_alpha = 0 # Numeratore di alpha
    num_beta = 0 # Numeratore di beta
    den = 0 # Base del denominatore di alpha e beta
    Di = D - (z_coord-z_coord.min())  # Distribuzione trasversale della profondita'
    N = Di.size # Numero di punti sulla trasversale

    # N punti trasversali -> N-1 intervalli (trapezi)
    for i in range( N-1 ): # Per ogni trapezio
        
        #    vertical stripe
        # 
        #         dy
        # 
        #        o-----o       <- water level
        #        |     |  
        #        |     |  DR
        #        |     |  
        #        |     o      zR     _ _
        #    DL  |    /       ^       |
        #        |   / dB     |       |
        #        |  /         |       |  dz
        #        | /\\ phi    |      _|_
        #    zL  o  ------    |       
        #    ^                |      
        #    |                |
        #    ------------------- z_coord=0
     
        yL, yR = y_coord[i], y_coord[i+1]
        zL, zR = z_coord[i], z_coord[i+1]
        DL, DR = Di[i], Di[i+1]
        dy = yR - yL
        dz = zR - zL
        dB = np.sqrt(dy**2+dz**2)
        cosphi = dy/dB
        # Geometric parameters:
        if DL<=0 and DR<=0:
            dy, dz = 0, 0
            DL, DR = 0, 0
        elif DL<0:
            dy = -dy*DR/dz
            dz = DR
            DL = 0
        elif DR<0:
            dy = dy*DL/dz
            dz = DL
            DR = 0
        
        #Metodo di Gauss:
        SUM = np.zeros(3)
        # TODO
        C = 0
        Dm = 0
        Phi=0
        teta1=0
        
        # Gauss weight loop
        for j in range(NG):
            Dm = (DR+DL)/2# + (DR-DL)/2*xj[j]
            # print(Dm)
            # print('tirante:', Dm, '   k:', k, '   ds:', ds)
            
            if Dm==0 or 2.5*np.log(11*Dm/(k*ds))<0:
                C=0
            else:
                C = 2.5*np.log(11*Dm/(k*ds))
            
            #den
            SUM[0] += wj[j]*C*Dm**(3/2)
            #num_alpha
            SUM[1] += wj[j]*C**(3)*Dm**(2.5)
            #num_beta
            SUM[2] += wj[j]*C**(2)*Dm**(2)
            
        den += dy/2*cosphi**(1/2)*SUM[0]
        num_alpha += dy/2*cosphi**(3/2)*SUM[1]
        num_beta += dy/2*cosphi*SUM[2]
        
        dOmega = (DR + DL)*dy/2
        
        #Calcolo di Omega: superficie della sezione
        Omega += dOmega
        
        #Calcolo di B: lunghezza del perimetro bagnato
        
        B += dB
     
        #Calcolo di b: larghezza della superficie libera
        b += dy
        
        #Calcolo di b: larghezza della superficie libera
        #Rh=Omega/B
        
        #Shields parameter
        teta_primo = (Dm*cosphi)*S/(delta*ds)
        array_teta = np.append(array_teta, teta_primo)
    
    
    count_active = np.count_nonzero(np.where(array_teta>=teta_c, 1, 0))        
        
        
    
    #Calcolo della portata Q
    Q = np.sqrt(S*g)*den
    
    #Calcolo della capacità di trasporto
    teta1 = (Omega/B)*S/(delta*ds)
    if teta1 >= teta_c:
        Qs = 8*(teta1-teta_c)**1.5*np.sqrt(9.81*delta*ds**3)*b
    else:
        Qs = 0
    # sumQs += qs
    Qs = sumQs
    
    #Condizione per procedere al calcolo anche quando il punto i è sommerso
    # mentre i+1 no.            
    if den==0:
        alpha = None
        beta = None
    else:
        alpha = Omega**2*(g*S)**(3/2)*num_alpha/den**3
        beta = Omega*g*S*num_beta/den**2
            
    return Q, Omega, b, B, alpha, beta, Qs, count_active



Dmax = z_coord.max()-z_coord.min()
Dmin = 0
i=0

# First guess value
D0 = (Dmax-Dmin)/2
Qn, Omega, b, B, alpha, beta, Qs, count_active = MotoUniforme(S, y_coord, z_coord, D0, NG, teta_c, ds)

Qmax, Omega, b, B, alpha, beta, Qs, count_active = MotoUniforme(S, y_coord, z_coord, Dmax, NG, teta_c, ds)
Qmin, Omega, b, B, alpha, beta, Qs, count_active = MotoUniforme(S, y_coord, z_coord, Dmin, NG, teta_c, ds)
if np.sign(Qmax-Q_target)==np.sign(Qmin-Q_target):
    print(' Soluntion out of boundaries')
else:
    # Check if h<h_min:
    while abs(Qn - Q_target)>toll:
        if i>max_iter:
            print('ERROR: max iterations reached!')
            break
        i+=1
        D0 = (Dmax+Dmin)/2
        Q0, Omega, b, B, alpha, beta, Qs, count_active = MotoUniforme(S, y_coord, z_coord, D0, NG, teta_c, ds)
        print(i)
        print(D0)
        print(Q0)
        if Q0>Q_target:
            Dmax=D0 # Update Dmax
        elif Q0<Q_target:
            Dmin=D0 # Update Dmin
        Qn=Q0
             


