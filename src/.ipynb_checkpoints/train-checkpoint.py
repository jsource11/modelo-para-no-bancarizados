# Código de Entrenamiento 
###########################

import pandas as pd
import xgboost as xgb
import pickle
import os

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline, make_pipeline
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.compose import ColumnTransformer

import statsmodels.api as sm # Statsmodels
from sklearn.model_selection import train_test_split #sklearn tiene gran cantidad de funciones para todo lo relacionado al
from sklearn.linear_model import Ridge

from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import roc_auc_score,roc_curve # ROC a diferencia del accuracy te da un valor justo de precision para datos desbalanceados


# Cargar la tabla transformada
def read_file_csv(filename):
    data_transacional = pd.read_csv(os.path.join('../data/processed', filename))
    print(filename, ' cargado correctamente')

    #explicitly create a placeholder for y-intercept: b0
    data_transacional['INTERCEPTO'] = 1

    # Reinicio del dataframe
    data_transacional = data_transacional.reset_index(drop=True)

    ### Normalización de variables númericas
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    data_transacional['LINEA_ASIG'] = scaler.fit_transform(data_transacional['LINEA_ASIG'].values.reshape(-1, 1))
    data_transacional['ingreso'] = scaler.fit_transform(data_transacional['ingreso'].values.reshape(-1, 1))
    data_transacional['antiguedad_laboral'] = scaler.fit_transform(data_transacional['antiguedad_laboral'].values.reshape(-1, 1))
    data_transacional['edad'] = scaler.fit_transform(data_transacional['edad'].values.reshape(-1, 1))
    data_transacional['frec_compra_spsa_12m'] = scaler.fit_transform(data_transacional['frec_compra_spsa_12m'].values.reshape(-1, 1))
    data_transacional['prom_imp_spsa_12m'] = scaler.fit_transform(data_transacional['prom_imp_spsa_12m'].values.reshape(-1, 1))
    data_transacional['frec_compra_far_12m'] = scaler.fit_transform(data_transacional['frec_compra_far_12m'].values.reshape(-1, 1))
    data_transacional['prom_imp_far_12m'] = scaler.fit_transform(data_transacional['prom_imp_far_12m'].values.reshape(-1, 1))
    data_transacional['frec_compra_oec_12m'] = scaler.fit_transform(data_transacional['frec_compra_oec_12m'].values.reshape(-1, 1))
    data_transacional['prom_imp_oec_12m'] = scaler.fit_transform(data_transacional['prom_imp_oec_12m'].values.reshape(-1, 1))
    data_transacional['frec_compra_pro_12m'] = scaler.fit_transform(data_transacional['frec_compra_pro_12m'].values.reshape(-1, 1))
    data_transacional['prom_imp_pro_12m'] = scaler.fit_transform(data_transacional['prom_imp_pro_12m'].values.reshape(-1, 1))

    data_X = ['LINEA_ASIG', 'antiguedad_laboral', 'edad', 'ingreso','frec_compra_spsa_12m', 'prom_imp_spsa_12m',
        'frec_compra_far_12m', 'prom_imp_far_12m', 'frec_compra_oec_12m', 'prom_imp_oec_12m', 'frec_compra_pro_12m', 'prom_imp_pro_12m',
        'estado_civil_encoding', 'sexo_encoding', 'grado_instruccion_encoding', 'tipotrabajador_encoding', 'RFM_Score_spsa', 'RFM_Score_far',
        'RFM_Score_oec', 'RFM_Score_pro']
    X = data_transacional[data_X]

    y = data_transacional[['desi_final_cda_encoding']]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3) # random state y seed metodo aleatorio simpl3


    logistic_reg = LogisticRegression()
    apply_classifier(logistic_reg,X_train, X_test, y_train, y_test)



def apply_classifier(clf,xTrain,xTest,yTrain,yTest):
    clf.fit(xTrain, yTrain) #Entrenamiento del modelo
    print('Modelo entrenado')
    predictions = clf.predict(xTest) #Validación sobre la data de testing
    conf_mtx = confusion_matrix(yTest,predictions) #Matriz de confusión de la data de testing real con la predicha

    # Guardamos el modelo entrenado para usarlo en produccion
    package = '../models/logisticregression_model.pkl'
    pickle.dump(clf, open(package, 'wb'))
    print('Modelo exportado correctamente en la carpeta models')



# Entrenamiento completo
def main():
    read_file_csv('data_train.csv')
    print('Finalizó el entrenamiento del Modelo')


if __name__ == "__main__":
    main()