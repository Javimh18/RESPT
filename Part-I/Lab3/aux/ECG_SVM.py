###     ###
#   SVM   #
###     ###



import numpy as np
import os
from sklearn import svm
import matplotlib.pyplot as plt
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import confusion_matrix , ConfusionMatrixDisplay
from sklearn.metrics import plot_det_curve
from sklearn.metrics import plot_roc_curve
from sklearn.metrics import det_curve, DetCurveDisplay
from sklearn.metrics import roc_curve



if __name__ == '__main__':
    path='C:\Users\manue\OneDrive\Documentos\DeepLearning_Uni\1er_Semestre\1_Review_Signal_Techniques\Part1\Lab1\Notebook3_ECG\ECGData'

    #
    ##
    ### Variables de los SVM ###
    ##   
    #

    C=1; KERNEL='poly'



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

    print('DATOS TRAIN')
    array_train = np.array(lista_train)    
    numero_datos, muestras, caract_locales = array_train.shape
    train = array_train.reshape(numero_datos,muestras*caract_locales)
    array_label_train = np.array(etiqueta_train)    
    



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
    modelo = svm.SVC(C=C,kernel=KERNEL,gamma=1,probability=True)
    modelo.fit(train, array_label_train)



    #
    ##
    ### Test ###
    ##   
    #

    print("TEST")
    y_train_pred = modelo.predict(train)
    y_test_pred = modelo.predict(test)
    probabilidades = modelo.predict_proba(test)
    print(np.shape(probabilidades))
    
    



