#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 14 11:10:00 2021

@author: erri
"""
# File containing all Engelund Gauss script's funcions

# Libreries
import numpy as np

#Set functiones variables:
g = 9.806

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


def MotoUniforme( iF, y, z, Y, NG, teta_c, ds):
    '''
    Calcola i parametri di moto uniforme per assegnato tirante

    Argomenti
    ---------

    iF: float
       pendenza del canale
    y: numpy.ndarray
      coordinate trasversali dei punti della sezione
    z: numpy.ndarray
      coordinate verticali dei punti della sezione
    Y: float
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
      portata alla quale si realizza la profondità Y di moto uniforme
    Omega: float
      area sezione bagnata alla profondita' Y
    b: float
      larghezza superficie libera alla profondita' Y
    alpha: float
      coefficiente di ragguaglio dell'energia alla profondita' Y
    beta: float
      coefficiente di ragguaglio della qdm alla profondita' Y
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
    Yi = Y - (z-z.min())  # Distribuzione trasversale della profondita'
    N = Yi.size # Numero di punti sulla trasversale

    # N punti trasversali -> N-1 intervalli (trapezi)
    for i in range( N-1 ): # Per ogni trapezio
        
        #    vertical stripe
        # 
        #         dy
        # 
        #        o-----o       <- water level
        #        |     |  
        #        |     |  YR
        #        |     |  
        #        |     o      zR     _ _
        #    YL  |    /       ^       |
        #        |   / dB     |       |
        #        |  /         |       |  dz
        #        | /\\ phi    |      _|_
        #    zL  o  ------    |       
        #    ^                |      
        #    |                |
        #    ------------------- z=0
     
        yL, yR = y[i], y[i+1]
        zL, zR = z[i], z[i+1]
        YL, YR = Yi[i], Yi[i+1]
        dy = yR - yL
        dz = zR - zL
        dB = np.sqrt(dy**2+dz**2)
        cosphi = dy/dB
        # Geometric parameters:
        if YL<=0 and YR<=0:
            dy, dz = 0, 0
            YL, YR = 0, 0
        elif YL<0:
            dy = -dy*YR/dz
            dz = YR
            YL = 0
        elif YR<0:
            dy = dy*YL/dz
            dz = YL
            YR = 0
        
        #Metodo di Gauss:
        SUM = np.zeros(3)
        # TODO
        C = 0
        Ym = 0
        Phi=0
        teta1=0
        
        # Gauss weight loop
        for j in range(NG):
            Ym = (YR+YL)/2# + (YR-YL)/2*xj[j]
            # print(Ym)
            # print('tirante:', Ym, '   k:', k, '   ds:', ds)
            
            if Ym==0 or 2.5*np.log(11*Ym/(k*ds))<0:
                C=0
            else:
                C = 2.5*np.log(11*Ym/(k*ds))
            
            #den
            SUM[0] += wj[j]*C*Ym**(3/2)
            #num_alpha
            SUM[1] += wj[j]*C**(3)*Ym**(2.5)
            #num_beta
            SUM[2] += wj[j]*C**(2)*Ym**(2)
            
        den += dy/2*cosphi**(1/2)*SUM[0]
        num_alpha += dy/2*cosphi**(3/2)*SUM[1]
        num_beta += dy/2*cosphi*SUM[2]
        
        dOmega = (YR + YL)*dy/2
        
        #Calcolo di Omega: superficie della sezione
        Omega += dOmega
        
        #Calcolo di B: lunghezza del perimetro bagnato
        
        B += dB
     
        #Calcolo di b: larghezza della superficie libera
        b += dy
        
        #Calcolo di b: larghezza della superficie libera
        #Rh=Omega/B
        
        #Shields parameter
        teta_primo = (Ym*cosphi)*iF/(delta*ds)
        array_teta = np.append(array_teta, teta_primo)
    
    
    count_active = np.count_nonzero(np.where(array_teta>=teta_c, 1, 0))        
        
        
    
    #Calcolo della portata Q
    Q = np.sqrt(iF*g)*den
    
    #Calcolo della capacità di trasporto
    teta1 = (Omega/B)*iF/(delta*ds)
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
        alpha = Omega**2*(g*iF)**(3/2)*num_alpha/den**3
        beta = Omega*g*iF*num_beta/den**2
            
    return Q, Omega, b, B, alpha, beta, Qs, count_active


def Critica( iF, y, z, Q, NG, MAXITER, tol, teta_c, ds ):
    '''
    Calcolo della profondita' critica ad assegnata portata

    Argomenti
    ---------
    iF: float
       pendenza del canale
    y: numpy.ndarray
      coordinate trasversali dei punti della sezione
    z: numpy.ndarray
      coordinate verticali dei punti della sezione
    Q: float
      portata
    MAXITER: int [default=100]
      numero massimo di iterazioni per l'algoritmo dicotomico
    tol: float
      tolleranza sull'eccesso di energia per l'algoritmo dicotomico
    NG: int [default=2]
      numero di punti di Gauss nel calcolo dei parametri

    Output
    ------
    Ym: float
       profondita' critica calcolata con il metodo dicotomico
    '''
    Ya = 1e-06
    Yb = z.max() - z.min()

    # Calcolo della profondita' critica con il metodo dicotomico
    # La funzione da annullare e' quella per l'eccesso di carico specifico sulla sezione
    
    if Energia( iF, y, z, Ya, Q, NG, teta_c, ds)*Energia( iF, y, z, Yb, Q, NG, teta_c, ds)<=0:
        for i in range(MAXITER):
            Ym = (Ya+Yb)/2
            Fa = Energia( iF, y, z, Ya, Q, NG, teta_c, ds)
            Fb = Energia( iF, y, z, Yb, Q, NG, teta_c, ds)
            Fm = Energia( iF, y, z, Ym, Q, NG, teta_c, ds)
            
            if np.abs(Fm)<tol:
                return Ym
            elif Fa*Fm<0:
                Yb = Ym
            else:
                Ya = Ym
                
        print('Maximum number of iteration reached!')
    else:
        print('Solution out of boundaries')
        
    return None


def Energia( iF, y, z, Y, Q, NG, teta_c, ds ):
    '''
    Eccesso di energia rispetto alla critica
    Funzione da annullare per trovare le condizioni critiche
    Argomenti
    ---------
    iF: float
       pendenza del canale
    y: numpy.ndarray
      coordinate trasversali dei punti della sezione
    z: numpy.ndarray
      coordinate verticali dei punti della sezione
    Y: float
      profondita'
    Q: float
      portata
    NG: int [default=2]
      numero di punti di Gauss nel calcolo dei parametri
    '''
    
    Q_temp, Omega, b, B, alpha, beta, Qs = MotoUniforme( iF, y, z, Y, NG, teta_c, ds)
    Fr2 = Q**2/(g*Omega**3/b)

    return alpha*Fr2 - 1