import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

#INTEGRACIÓN Y FUSIÓN DE DATOS
# Definir la ruta relativa a la carpeta que contiene los datasets
data_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Dataset")

# Crear rutas completas para cada archivo CSV
orders_path    = os.path.join(data_path, "olist_orders_dataset.csv")
customers_path = os.path.join(data_path, "olist_customers_dataset.csv")
reviews_path   = os.path.join(data_path, "olist_order_reviews_dataset.csv")
# products_path no se usará en este ejemplo

# Cargar archivos CSV
orders    = pd.read_csv(orders_path)
customers = pd.read_csv(customers_path)
reviews   = pd.read_csv(reviews_path)

# Visualiza los tamaños de cada dataset antes de la unión para verificar consistencia
print("Órdenes:", orders.shape)
print("Clientes:", customers.shape)
print("Reseñas:", reviews.shape)

# Realizar merge utilizando llaves comunes:
# Se asume que:
#   - 'customer_id' es la llave que une orders con customers.
#   - 'order_id' es la llave que une orders con reviews.
df_merged = orders.merge(customers, how='left', on='customer_id') \
                  .merge(reviews, how='left', on='order_id')

# Verificar la consistencia de la fusión comparando el número total de registros
print("Shape después de merge:", df_merged.shape)

# LIMPIEZA DE DATOS
# Tratamiento de Valores Faltantes
print("\nValores nulos por columna:")
print(df_merged.isnull().sum())

# Reemplazar valores nulos usando forward fill con el método ffill() directamente
df_merged.ffill(inplace=True)

# Normalización de Formatos
if 'order_date' in df_merged.columns:
    df_merged['order_date'] = pd.to_datetime(df_merged['order_date'], errors='coerce')

if 'price' in df_merged.columns:
    df_merged['price'] = pd.to_numeric(df_merged['price'], errors='coerce')

# Eliminación de Duplicados
df_merged.drop_duplicates(inplace=True)
print("Shape después de eliminar duplicados:", df_merged.shape)

# TRANSFORMACIÓN Y NORMALIZACIÓN
# Codificación de Variables Categóricas
if 'category_column' in df_merged.columns:  
    df_encoded = pd.get_dummies(df_merged, columns=['category_column'], drop_first=True)
else:
    df_encoded = df_merged.copy()

#Escalado de Variables Numéricas
numeric_cols = []
for col in ['price', 'delivery_time']:  
    if col in df_encoded.columns:
        numeric_cols.append(col)

if numeric_cols:
    scaler = StandardScaler()
    df_encoded[numeric_cols] = scaler.fit_transform(df_encoded[numeric_cols])
    print(f"\nVariables escaladas: {numeric_cols}")


# VALIDACIÓN DEL DATASET PREPROCESADO Y GUARDADO
print("\nResumen final del dataset preprocesado:")
print(df_encoded.info())
print(df_encoded.head())

# Guardar el dataset preprocesado para su uso en modelado
output_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "dataset_preprocesado.csv")
df_encoded.to_csv(output_path, index=False)
print(f"\nEl dataset preprocesado ha sido guardado en: {output_path}")
