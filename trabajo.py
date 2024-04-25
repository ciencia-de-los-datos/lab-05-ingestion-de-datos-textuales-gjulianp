import os
import pandas as pd

def load_data(directorio):
    datos = []
    for categoria in os.listdir(directorio):
        ruta_categoria = os.path.join(directorio, categoria)
        if os.path.isdir(ruta_categoria):
            for archivo in os.listdir(ruta_categoria):
                ruta_archivo = os.path.join(ruta_categoria, archivo)
                with open(ruta_archivo, 'r', encoding='latin1') as f:
                    frase = f.read().strip()
                    datos.append((frase, categoria))  # Agregar el nombre de la categoría como el sentimiento
    return datos


def remove_na_rows(datos):
    """Elimina las filas con valores nulos en las columnas 'phrase' y 'sentiment'"""
    datos = datos.dropna(subset=['phrase', 'sentiment'])
    datos = datos[~datos['phrase'].str.contains("Bud1")]
    return datos

def main():
    """Función principal"""
    # Procesar datos de entrenamiento
    train_data = load_data('data/train')

    # Procesar datos de prueba
    test_data = load_data('data/test')

    # Crear DataFrames con Pandas
    train_df = pd.DataFrame(train_data, columns=['phrase', 'sentiment'])
    test_df = pd.DataFrame(test_data, columns=['phrase', 'sentiment'])

    # Eliminar filas con valores nulos
    train_df = remove_na_rows(train_df)
    test_df = remove_na_rows(test_df)

    # Contar los valores únicos en la columna 'sentiment'
    train_sentiment_counts = train_df['sentiment'].value_counts()
    test_sentiment_counts = test_df['sentiment'].value_counts()

    # Guardar DataFrames en archivos CSV
    train_df.to_csv('train_dataset.csv', index=False)
    test_df.to_csv('test_dataset.csv', index=False)

    print("Archivos CSV generados exitosamente: 'train_dataset.csv' y 'test_dataset.csv'")
    print("Conteo de sentimientos en datos de entrenamiento:")
    print(train_sentiment_counts)
    print("Conteo de sentimientos en datos de prueba:")
    print(test_sentiment_counts)

if __name__ == "__main__":
    main()

