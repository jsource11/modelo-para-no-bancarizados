# Script de Preparación de Datos
###################################

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import missingno as msno
import os
from sklearn.impute import KNNImputer


# Leemos los archivos csv
def read_file_csv(filename):
    # Display all columns and rows in the output
    pd.set_option("display.max_columns",None)
    pd.set_option('display.max_rows',None)
    df = pd.read_csv(os.path.join('../data/raw/', filename))
    print(filename, ' cargado correctamente')
    return df


# Realizamos la transformación de datos
def data_preparation(data_transacional):
    
    ##### LIMPIEZA DE DATOS #####

    ## Eliminación de columnas con mas del 70% de registros null
    null_percentage = data_transacional.isnull().sum() / len(data_transacional)
    threshold = 0.7
    columns_to_drop = null_percentage[null_percentage >= threshold].index
    data_transacional.drop(columns=columns_to_drop, inplace=True)

    ## Eliminación de las variables _6m_ que representan los 6 meses
    data_transacional = data_transacional.loc[:, ~data_transacional.columns.str.contains('_6m_')]
    ## Eliminación de las variables 6m que representan los 6 meses
    data_transacional = data_transacional.loc[:, ~data_transacional.columns.str.contains('_6m')]

    ## Filtramos todos los campos que contengan _12m
    df_12m = data_transacional.filter(like='_12m', axis=1)
    df_12m = df_12m[[ 'max_fec_compra_spsa_12m', 'frec_compra_spsa_12m', 'prom_imp_spsa_12m','max_fec_compra_far_12m', 'frec_compra_far_12m', 'prom_imp_far_12m','max_fec_compra_oec_12m', 'frec_compra_oec_12m', 'prom_imp_oec_12m','max_fec_compra_pro_12m', 'frec_compra_pro_12m', 'prom_imp_pro_12m']]
    ### Eliminamos del dataframe principal las columnas que contengan el _12m que son los 12 meses
    data_transacional = data_transacional.loc[:, ~data_transacional.columns.str.contains('_12m')]

    ## Eliminación de columnas a no utilizar
    ## columnas_eliminar = ['APPID','appid_1','id_doc_1','antig_ruc_solici', 'PERID', 'deuda_cm', 'ingreso_calc','ingreso_estimado' , 'ingreso_neto_mensual', 'nivel_estudios', 'nacionalidad', 'nombre_empresa', 'num_dependientes','profesion_cargo', 'profesion_ocupacion', 'saldo_cm', 'tipo_actividad','tipo_empleador', 'tipo_ingreso', 'tipo_vivienda', 'n_imp_tot_spsa', 'n_imp_tot_far', 'n_imp_efe_spsa', 'n_imp_efe_far', 'n_imp_cre_spsa','n_imp_cre_far', 'n_imp_deb_spsa', 'n_imp_deb_far', 'n_imp_agora_spsa', 'n_imp_agora_far', 'n_imp_otros_spsa', 'n_imp_otros_far','pct_com_efe_spsa', 'pct_com_efe_far', 'pct_com_cre_spsa', 'pct_com_cre_far', 'pct_com_deb_spsa', 'pct_com_deb_far','pct_com_agora_spsa', 'pct_com_agora_far', 'pct_com_otros_spsa', 'pct_com_otros_far', 'max_compra_spsa', 'max_compra_far','max_fec_compra_spsa', 'max_fec_compra_far', 'num_compras_spsa', 'num_compras_far', 'fecha', 'id_doc_2', 'nro_trx_tot', 'nro_trx_toh','pct_imp_toh', 'pct_imp_vea_toh', 'pct_imp_far_toh', 'rtc_prom_imp_tot_vea_03_03', 'rtc_max_imp_tot_vea_01_05','rtc_prom_imp_tot_far_03_03', 'rtc_max_imp_tot_far_01_05', 'rtc_max_imp_tot_vea_01_02', 'rtc_max_imp_toh_vea_01_02','rtc_max_imp_tot_far_01_02', 'rtc_max_imp_miscelaneos_01_02', 'rtc_prom_imp_medicamento_etico_03_03','rtc_max_imp_medicamento_etico_01_05','rtc_max_imp_medicamento_etico_01_02', 'rtc_prom_imp_frescos_03_03','rtc_max_imp_frescos_01_02', 'rtc_prom_imp_consumo_03_03', 'rtc_max_imp_consumo_01_02', 'rtc_prom_imp_abarrotes_03_03','rtc_max_imp_abarrotes_01_05', 'rtc_max_imp_abarrotes_01_02', 'departamento', 'provincia', 'distrito', 'fecha_nac', 'ubigeo','grado_instruccion_grupo', 'ingreso_1', 'situacion', 'tipodocumento']
    columnas_eliminar = ['APPID','appid_1','id_doc_1','antig_ruc_solici', 'PERID', 'deuda_cm', 'ingreso_calc','ingreso_estimado' , 'ingreso_neto_mensual', 'nivel_estudios', 'nacionalidad', 'nombre_empresa', 'num_dependientes','profesion_cargo', 'profesion_ocupacion', 'saldo_cm', 'tipo_actividad','tipo_empleador', 'tipo_ingreso', 'tipo_vivienda', 'n_imp_tot_spsa', 'n_imp_tot_far', 'n_imp_efe_spsa', 'n_imp_efe_far', 'n_imp_cre_spsa','n_imp_cre_far', 'n_imp_deb_spsa', 'n_imp_deb_far', 'n_imp_agora_spsa', 'n_imp_agora_far', 'n_imp_otros_spsa', 'n_imp_otros_far','pct_com_efe_spsa', 'pct_com_efe_far', 'pct_com_cre_spsa', 'pct_com_cre_far', 'pct_com_deb_spsa', 'pct_com_deb_far','pct_com_agora_spsa', 'pct_com_agora_far', 'pct_com_otros_spsa', 'pct_com_otros_far', 'max_compra_spsa', 'max_compra_far','max_fec_compra_spsa', 'max_fec_compra_far', 'num_compras_spsa', 'num_compras_far', 'fecha', 'id_doc_2', 'nro_trx_tot', 'nro_trx_toh','pct_imp_toh', 'pct_imp_vea_toh', 'pct_imp_far_toh', 'rtc_prom_imp_tot_vea_03_03', 'rtc_max_imp_tot_vea_01_05','rtc_prom_imp_tot_far_03_03', 'rtc_max_imp_tot_far_01_05', 'rtc_max_imp_tot_vea_01_02', 'rtc_max_imp_toh_vea_01_02','rtc_max_imp_tot_far_01_02', 'rtc_prom_imp_medicamento_etico_03_03','rtc_max_imp_medicamento_etico_01_05','rtc_max_imp_medicamento_etico_01_02','rtc_max_imp_frescos_01_02', 'rtc_prom_imp_consumo_03_03', 'rtc_max_imp_consumo_01_02', 'rtc_prom_imp_abarrotes_03_03','rtc_max_imp_abarrotes_01_05', 'rtc_max_imp_abarrotes_01_02', 'departamento', 'provincia', 'distrito', 'fecha_nac', 'ubigeo','grado_instruccion_grupo', 'ingreso_1', 'situacion', 'tipodocumento']
    data_transacional = data_transacional.drop(columnas_eliminar, axis=1)

    # Unión de los dos dataframe los datos generales y las transacionales
    data_transacional = pd.merge(data_transacional, df_12m, left_index=True, right_index=True)


    # Se procede a elimanar los registros
    data_transacional = data_transacional.drop_duplicates()

    ## Eliminaremos todos los registros null en el campo edad
    data_transacional = data_transacional.dropna(subset=['edad'])

    ## Procedemos a reemplazar los valores NaN con la categoria mas común que es 'EMPLEADO' de la variable tipotrabajador
    data_transacional['tipotrabajador'] = data_transacional['tipotrabajador'].fillna(data_transacional['tipotrabajador'].mode()[0])

    ## Procedemos a reemplazar los valores NaN con la categoria mas común que es 'SECUNDARIA COMPLETA' de la variable grado_instruccion
    data_transacional['grado_instruccion'] = data_transacional['grado_instruccion'].fillna(data_transacional['grado_instruccion'].mode()[0])

    ### Procedemos a transformar la variable max_fec_compra_spsa_12m a tipo de dato datetime
    data_transacional['max_fec_compra_spsa_12m'] = pd.to_datetime(data_transacional['max_fec_compra_spsa_12m'])

    ### reemplazamos con la mediana a la columna max_fec_compra_spsa_12m con el valor de 2023-05-12
    data_transacional['max_fec_compra_spsa_12m'].fillna(data_transacional['max_fec_compra_spsa_12m'].median(), inplace=True)

    ### Procedemos a transformar la variable max_fec_compra_spsa_12m a tipo de dato datetime
    data_transacional['max_fec_compra_oec_12m'] = pd.to_datetime(data_transacional['max_fec_compra_oec_12m'])

    ### reemplazamos con la moda a la columna max_fec_compra_oec_12m con el valor de 2023-04-01
    data_transacional['max_fec_compra_oec_12m'].fillna(data_transacional['max_fec_compra_oec_12m'].mode()[0], inplace=True)

    ### Procedemos a transformar la variable max_fec_compra_spsa_12m a tipo de dato datetime
    data_transacional['max_fec_compra_pro_12m'] = pd.to_datetime(data_transacional['max_fec_compra_pro_12m'])

    # Interpolar los valores faltantes en la columna
    data_transacional['max_fec_compra_pro_12m'] = data_transacional['max_fec_compra_pro_12m'].interpolate()

    ### Verificamos la media de la variable
    media = data_transacional['prom_imp_pro_12m'].mean()
    ### Remplazamos los valores NaN con la media 262.45626414651537
    data_transacional['prom_imp_pro_12m'] = data_transacional['prom_imp_pro_12m'].fillna(media)

    ### Verificamos la media de la variable
    media = data_transacional['prom_imp_oec_12m'].mean()
    ### Remplazamos los valores NaN con la media 357.3047146961781
    data_transacional['prom_imp_oec_12m'] = data_transacional['prom_imp_oec_12m'].fillna(media)

    # Reamplazamos los valores unicos
    data_transacional = data_transacional.replace(" ", np.nan)
    # Eliminamos los valores nulos
    data_transacional = data_transacional.dropna()


    #### TRANSFORMACIÓN DE TIPOS DE DATOS ####

    ### Cambiando el tipo de datos de las variables
    data_transacional['antiguedad_laboral'] =  data_transacional['antiguedad_laboral'].astype('int')
    data_transacional['edad'] =  data_transacional['edad'].astype('int')
    data_transacional['ingreso'] =  data_transacional['ingreso'].astype('int')
    data_transacional['LINEA_ASIG'] =  data_transacional['LINEA_ASIG'].astype('int')
    data_transacional['flag_nobancariza'] =  data_transacional['flag_nobancariza'].astype('object')
    data_transacional['max_fec_compra_far_12m'] = pd.to_datetime(data_transacional['max_fec_compra_far_12m'])

    ### Cambiando el tipo de dato sexo por Imputación KNN
    mapping =  {'M':1, 'F':2, 'X':3}
    data_transacional["sexo"] = data_transacional["sexo"].map(mapping)
    data_transacional["sexo"].unique()

    imputer = KNNImputer(n_neighbors=2)
    df_imputed = data_transacional.copy()
    # Reemplazar los valores 3 por NaN temporalmente
    df_imputed.loc[df_imputed['sexo'] == 3, 'sexo'] = np.nan
    # Imputación KNN
    df_imputed[['sexo']] = imputer.fit_transform(df_imputed[['sexo']])
    df_imputed['sexo'] = df_imputed['sexo'].round().astype(int)
    #df_imputed.loc[~df_imputed['sexo'].isin([1, 2]), 'sexo'] = 1  # O forzar a 2 si se prefiere
    data_transacional['sexo'] = df_imputed['sexo']


    #### Segmentación de la data de flag_nobancariza ####

    # Creación del dataframe de los registros no bancarizados para las prubas
    data_transacional_no_bancarizado = data_transacional[data_transacional['flag_nobancariza']==1]

    # Creación del dataframe de los registros bancarizados para la realización del estudio
    data_transacional = data_transacional[data_transacional['flag_nobancariza']==0]

    #### AD / AU ####
    #Procedemos a la eliminación de los registros de antiguedad_laboral menores a
    data_transacional = data_transacional.drop(data_transacional[data_transacional['antiguedad_laboral'] < 0 ].index)

    #Procedemos a la eliminación de los registros de antiguedad_laboral menores a
    data_transacional = data_transacional.drop(data_transacional[data_transacional['prom_imp_spsa_12m'] < 0 ].index)

    #Procedemos a la eliminación de los registros de antiguedad_laboral menores a
    data_transacional = data_transacional.drop(data_transacional[data_transacional['prom_imp_oec_12m'] < 0 ].index)


    from sklearn.preprocessing import LabelEncoder
    encoder = LabelEncoder()
    data_transacional['desi_final_cda_encoding'] = encoder.fit_transform(data_transacional['desi_final_cda'])
    data_transacional['estado_civil_encoding'] = encoder.fit_transform(data_transacional['estado_civil'])
    data_transacional['sexo_encoding'] = encoder.fit_transform(data_transacional['sexo'])
    data_transacional['grado_instruccion_encoding'] = encoder.fit_transform(data_transacional['grado_instruccion'])
    data_transacional['tipotrabajador_encoding'] = encoder.fit_transform(data_transacional['tipotrabajador'])

    # Copia del df para las ección de RFM
    data_RFM = data_transacional.copy()

    # Segmentación RFM

    #Elección de variables del dataframe transacional
    data_RFM_df = data_RFM[['ID_DOC','max_fec_compra_spsa_12m','frec_compra_spsa_12m','prom_imp_spsa_12m','max_fec_compra_far_12m','frec_compra_far_12m','prom_imp_far_12m','max_fec_compra_oec_12m','frec_compra_oec_12m','prom_imp_oec_12m','max_fec_compra_pro_12m','frec_compra_pro_12m','prom_imp_pro_12m']]


    #Calculando la recencia
    df_recency = data_RFM_df.groupby('ID_DOC').agg({'max_fec_compra_spsa_12m': 'max','max_fec_compra_far_12m': 'max','max_fec_compra_oec_12m': 'max','max_fec_compra_pro_12m': 'max'}).reset_index()

    for column in data_RFM_df.columns:
        if column.startswith('max_fec_compra'):
            max_date = data_RFM_df[column].max()
            prefix = column.split('_')[3]  # Obtener el prefijo de la columna
            # Recency
            columna_ultima_compra = 'ultima_compra_' + prefix
            df_recency[columna_ultima_compra] = max_date
            df_recency[columna_ultima_compra] = pd.to_datetime(df_recency[columna_ultima_compra])
            data_RFM_df[column] = pd.to_datetime(data_RFM_df[column])
            columna_recency = 'R_' + prefix
            df_recency[columna_recency] = (df_recency[columna_ultima_compra] - df_recency[column]).dt.days

    #Calculando la frecuencia

    df_frequency = data_transacional.groupby('ID_DOC').agg({'frec_compra_spsa_12m': 'sum','frec_compra_far_12m': 'sum','frec_compra_oec_12m': 'sum','frec_compra_pro_12m': 'sum'}).reset_index()
    df_frequency = df_frequency.rename(columns={'frec_compra_spsa_12m': 'F_spsa', 'frec_compra_far_12m': 'F_far', 'frec_compra_oec_12m': 'F_oec', 'frec_compra_pro_12m': 'F_pro'})


    #Calculando el valor monetario

    df_monetary = data_transacional.groupby('ID_DOC').agg({
        'prom_imp_spsa_12m': 'sum',
        'prom_imp_far_12m': 'sum',
        'prom_imp_oec_12m': 'sum',
        'prom_imp_pro_12m': 'sum'
    }).reset_index()

    df_monetary = df_monetary.rename(columns={'prom_imp_spsa_12m': 'M_spsa', 'prom_imp_far_12m': 'M_far', 'prom_imp_oec_12m': 'M_oec', 'prom_imp_pro_12m': 'M_pro'})


    #Uniendo los dataframes df_recency, df_frequency y df_monetary

    rf_df = df_recency[['ID_DOC', 'R_spsa', 'R_far', 'R_oec', 'R_pro']].merge(df_frequency, on='ID_DOC')
    rfm_df = rf_df.merge(df_monetary, on='ID_DOC')
    rfm_df.head()

    #Data para RFM Global
    rfm_df_global = rfm_df.copy()


    #Rankiando el ID_DOC en base al RFM

    #Crear nuevas columnas con rank
    for col in rfm_df.columns[1:]:
        if col.find('_', 2):
            #Crear nuevas columnas con rank
            rfm_df[col + '_rank'] = rfm_df[col].rank(ascending=False)
            #Crear nuevas columnas con rank normalizado
            rank_col = col + '_rank'
            rfm_df[col + '_rank_norm'] = (rfm_df[rank_col] / rfm_df[rank_col].max()) * 100

    #Eliminar las columnas
    cols_to_drop = [col for col in rfm_df.columns if col.endswith('_rank')]
    rfm_df.drop(cols_to_drop, axis=1, inplace=True)

    #Calculating la puntuación RFM

    #Creación de las columnas RFM por cada comercio
    for c in ['spsa', 'far', 'oec', 'pro']:
        rfm_df[f'RFM_Score_{c}'] = 0.15 * rfm_df[f'R_{c}_rank_norm'] + 0.28 * rfm_df[f'F_{c}_rank_norm'] + 0.57 * rfm_df[f'M_{c}_rank_norm']
        rfm_df[f'RFM_Score_{c}'] *= 0.05

    RFM_Score = rfm_df[['ID_DOC', 'RFM_Score_spsa', 'RFM_Score_far', 'RFM_Score_oec', 'RFM_Score_pro']]

    df_rfm_g = rfm_df[['ID_DOC','RFM_Score_spsa','RFM_Score_far','RFM_Score_oec','RFM_Score_pro']]

    data_transacional = data_transacional.merge(df_rfm_g , on='ID_DOC')


    data_t_v = ['LINEA_ASIG', 'antiguedad_laboral', 'edad', 'ingreso','frec_compra_spsa_12m', 'prom_imp_spsa_12m',
        'frec_compra_far_12m', 'prom_imp_far_12m', 'frec_compra_oec_12m', 'prom_imp_oec_12m', 'frec_compra_pro_12m', 'prom_imp_pro_12m',
        'estado_civil_encoding', 'sexo_encoding', 'grado_instruccion_encoding', 'tipotrabajador_encoding', 'RFM_Score_spsa', 'RFM_Score_far',
        'RFM_Score_oec', 'RFM_Score_pro']

    #Segmentación en base a la puntuación RFM --> 3 Segmentos

    for column in ['spsa', 'far', 'oec', 'pro']:
        RFM_Score[f"Customer_segment_{column}"] = np.where(RFM_Score[f'RFM_Score_{column}'] > 3.5, "Oro",(np.where(RFM_Score[f'RFM_Score_{column}'] > 2, "Plata",('Bronce'))))

    #Ordenando el dataframe
    RFM_Score = RFM_Score[['ID_DOC', 'RFM_Score_spsa', 'Customer_segment_spsa', 'RFM_Score_far', 'Customer_segment_far', 'RFM_Score_oec', 'Customer_segment_oec', 'RFM_Score_pro', 'Customer_segment_pro']]


    print('Transformación de datos completa')
    return data_transacional


# Exportamos la matriz de datos con las columnas seleccionadas
def data_exporting(df, features, filename):
    dfp = df[features]
    dfp.to_csv(os.path.join('../data/processed/', filename))
    print(filename, 'exportado correctamente en la carpeta processed')


# Generamos las matrices de datos que se necesitan para la implementación

def main():
    # Matriz de Entrenamiento
    df1 = read_file_csv('DATA_MUESTRA.csv')
    tdf1 = data_preparation(df1)
    data_exporting(tdf1,['ID_DOC', 'LINEA_ASIG', 'desi_final_cda', 'antiguedad_laboral', 'edad','estado_civil', 'flag_nobancariza', 'ingreso', 'sexo','grado_instruccion', 'tipotrabajador', 'max_fec_compra_spsa_12m','frec_compra_spsa_12m', 'prom_imp_spsa_12m', 'max_fec_compra_far_12m','frec_compra_far_12m', 'prom_imp_far_12m', 'max_fec_compra_oec_12m','frec_compra_oec_12m', 'prom_imp_oec_12m', 'max_fec_compra_pro_12m','frec_compra_pro_12m', 'prom_imp_pro_12m', 'desi_final_cda_encoding','estado_civil_encoding', 'sexo_encoding', 'grado_instruccion_encoding','tipotrabajador_encoding', 'RFM_Score_spsa', 'RFM_Score_far','RFM_Score_oec', 'RFM_Score_pro'],'data_train.csv')
    
    # Matriz de Validación
    df2 = read_file_csv('DATA_MUESTRA_NUEVO.csv')
    tdf2 = data_preparation(df2)
    data_exporting(tdf2, ['ID_DOC', 'LINEA_ASIG', 'desi_final_cda', 'antiguedad_laboral', 'edad','estado_civil', 'flag_nobancariza', 'ingreso', 'sexo','grado_instruccion', 'tipotrabajador', 'max_fec_compra_spsa_12m','frec_compra_spsa_12m', 'prom_imp_spsa_12m', 'max_fec_compra_far_12m','frec_compra_far_12m', 'prom_imp_far_12m', 'max_fec_compra_oec_12m','frec_compra_oec_12m', 'prom_imp_oec_12m', 'max_fec_compra_pro_12m','frec_compra_pro_12m', 'prom_imp_pro_12m', 'desi_final_cda_encoding','estado_civil_encoding', 'sexo_encoding', 'grado_instruccion_encoding','tipotrabajador_encoding', 'RFM_Score_spsa', 'RFM_Score_far','RFM_Score_oec', 'RFM_Score_pro'],'data_val.csv')
    
    # Matriz de Scoring
    #df3 = read_file_csv('WA_Fn-UseC_-xx.csv')
    #tdf3 = data_preparation(df3)
    #data_exporting(tdf3, [xx],'xx.csv')
    
    
if __name__ == "__main__":
    main()