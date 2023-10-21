###     ###
#   GMM   #
###     ###


import numpy as np
import os
#import pandas as pd
from sklearn.mixture import GaussianMixture
from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import det_curve
from sklearn.metrics import DetCurveDisplay
import matplotlib.pyplot as plt

if __name__ == '__main__':
    path='C:\Users\manue\OneDrive\Documentos\DeepLearning_Uni\1er_Semestre\1_Review_Signal_Techniques\Part1\Lab1\Notebook3_ECG\ECGData'

    #
    ##
    ### Variables de los GMM ###
    ##   
    #

    n=5; forma='diag'



    #
    ##
    ### División de datos ###
    ##   
    #

    print("DIVISIÓN DE LOS DATOS")
    lista_train = []
    etiqueta_train = []
    lista_test = []
    etiqueta_test = []




    #
    ##
    ### TRAIN ###
    ##   
    #

    print("DATOS TRAIN")
    array_train = np.array(lista_train)
    numero_datos, muestras, caract_locales = array_train.shape
    train = array_train.reshape(numero_datos,muestras*caract_locales)
  
    array_label_train = np.array(etiqueta_train)  

    train1=[]; train2=[]; train3=[]; 
    array_label_train1=[]; array_label_train2=[]; array_label_train3=[]; 
    
    for i in range(len(array_label_train)):     
        if array_label_train[i] == 1:
            train1.append(train[i])
            array_label_train1.append(1)
        if array_label_train[i] == 2:
            train2.append(train[i])
            array_label_train2.append(2)
        if array_label_train[i] == 3:
            train3.append(train[i])
            array_label_train3.append(3)



    #
    ##
    ### TEST ###
    ##   
    #

    print("DATOS TEST")
    array_test = np.array(lista_test)
    numero_datos2, muestras2, caract_locales2 = array_test.shape
    test = array_test.reshape(numero_datos2,muestras2*caract_locales2)
    
    array_label_test = np.array(etiqueta_test)



    #
    ##
    ### Entrenamiento ###
    ##   
    #
    
    print("ENTRENAMIENTO")
    modelo1 = GaussianMixture(n_components=n , covariance_type=forma, random_state=0)
    modelo1.fit(train1)

    modelo2 = GaussianMixture(n_components=n , covariance_type=forma, random_state=0)
    modelo2.fit(train2)

    modelo3 = GaussianMixture(n_components=n , covariance_type=forma, random_state=0)
    modelo3.fit(train3)



    #
    ##
    ### Test ###
    ##   
    #

    print("TEST")
    acierto=0
    fallo=0
    #print(test[0]); print(test[1])
    lista =[]; lista1 =[]; lista2 =[]; lista3 =[]; 
    listas1=[]; listas2=[]; listas3=[]; 


    for i in range(len(array_label_test)):
        #test[i]= np.reshape(test[i],(1,-1))
        score1 = modelo1.score(np.reshape(test[i],(1,-1)), array_label_test[i])
        #score_train1 = modelo1.score(train1, array_label_train1)
        #print("Score del 1 TEST: ", score1, "  Score del 1 TRAIN: ", score_train1)
        score2 = modelo2.score(np.reshape(test[i],(1,-1)), array_label_test[i])
        #score_train2 = modelo2.score(train2, array_label_train2)
        #print("Score del 2 TEST: ", score2, "  Score del 2 TRAIN: ", score_train2)
        score3 = modelo3.score(np.reshape(test[i],(1,-1)), array_label_test[i])
        #score_train3 = modelo3.score(train3, array_label_train3)
        #print("Score del 3 TEST: ", score3, "  Score del 3 TRAIN: ", score_train3)

        puntuacion = np.zeros(3)
        puntuacion[1]=score1; puntuacion[2]=score2; puntuacion[3]=score3; 
        max_punt = np.max(puntuacion)
        posicion = np.where(puntuacion == max_punt)
        lista.append(posicion[0][0])    

        if (posicion==array_label_test[i]):
            #print("ACIERTO, puntuacion: ", max_punt)
            acierto+=1      

            if (array_label_test[i]==1):
                lista1.append(1)
                listas1.append(score1)
  
            if (array_label_test[i]==2):
                lista2.append(1)
                listas2.append(score2)

            if (array_label_test[i]==3):
                lista3.append(1)
                listas3.append(score3)  
            
        else:
            #print("NO ACIERTO")
            #print('CUAL ES:', array_label_test[i], 'CUAL NO ES:', posicion[0])
            fallo+=1   

            if (array_label_test[i]==1):
                lista1.append(0)
                listas1.append(score1)
  
            if (array_label_test[i]==2):
                lista2.append(0)
                listas2.append(score2)

            if (array_label_test[i]==3):
                lista3.append(0)
                listas3.append(score3)
          
            
           

    array = np.array(lista)
    C = confusion_matrix(array_label_test,array)
    print('Porcentaje correctos Test:', 100*acierto/(acierto + fallo),'%') 
    print('Matriz de Confusion: \n',C)  
    cm = ConfusionMatrixDisplay(C)

    D1= det_curve(np.array(lista1),np.array(listas1))
    det1 = DetCurveDisplay( fpr=D1[0], fnr=D1[1], estimator_name="DET 1")

    D2= det_curve(np.array(lista2),np.array(listas2))
    det2 = DetCurveDisplay( fpr=D2[0], fnr=D2[1], estimator_name="DET 2")

    D3= det_curve(np.array(lista3),np.array(listas3))
    det3 = DetCurveDisplay( fpr=D3[0], fnr=D3[1], estimator_name="DET 3")
  

    print('FINAL')

    #PARA HACER LAS DET POR DÍTITO SEPARADO

    #det0.plot()
    #det1.plot()
    #det2.plot()
    #det3.plot()
    #det4.plot()
    #det5.plot()
    #det6.plot()
    #det7.plot()
    #det8.plot()
    #det9.plot()

    #PARA HACER TODAS A LA VEZ

    plt.plot(D1[0],D1[1],label='1')
    plt.plot(D2[0],D2[1],label='2')
    plt.plot(D3[0],D3[1],label='3')
    plt.plot([0, 1], [0, 1], 'k--')  # curva y=x
    plt.xlabel('Tasa de falsos positivos')
    plt.ylabel('Tasa de falsos negativos')
    plt.legend()
    plt.show()