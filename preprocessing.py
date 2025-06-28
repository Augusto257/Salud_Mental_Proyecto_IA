import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import LabelEncoder
from pathlib import Path
import sys
import json

# --- Rutas a los LabelEncoders y al Modelo ---
MODELS_DIR = Path('models')

# Cargar metadatos para obtener nombres de características
METADATA_PATH = MODELS_DIR / 'model_metadata.json'
try:
    with open(METADATA_PATH, 'r') as f:
        MODEL_METADATA = json.load(f)
    GLOBAL_MODEL_FEATURES = MODEL_METADATA.get("feature_names")
    print("✅ Metadatos del modelo cargados correctamente.")
except FileNotFoundError:
    print(f"❌ Error: Archivo de metadatos del modelo no encontrado en {METADATA_PATH}.", file=sys.stderr)
    print("Asegúrate de haber guardado 'model_metadata.json' en Colab y colocado en la carpeta 'models'.", file=sys.stderr)
    sys.exit(1)
except Exception as e:
    print(f"❌ Error cargando metadatos del modelo: {str(e)}", file=sys.stderr)
    sys.exit(1)

if GLOBAL_MODEL_FEATURES is None:
    print("❌ Error: 'feature_names' no se encontró en 'model_metadata.json'.", file=sys.stderr)
    sys.exit(1)


# Diccionario para almacenar todos los LabelEncoders cargados
LOADED_LABEL_ENCODERS = {}
# Columnas que son categóricas y necesitan LabelEncoding
CATEGORICAL_COLS_TO_ENCODE = [
    'Gender', 'Country', 'self_employed', 'family_history',
    'treatment', 'work_interfere', 'no_employees', 'remote_work',
    'tech_company', 'benefits', 'care_options', 'wellness_program',
    'seek_help', 'anonymity', 'leave', 'mental_health_consequence',
    'phys_health_consequence', 'coworkers', 'supervisor',
    'mental_health_interview', 'phys_health_interview',
    'mental_vs_physical', 'obs_consequence'
]

# Cargar todos los LabelEncoders individualmente
for col in CATEGORICAL_COLS_TO_ENCODE:
    encoder_path = MODELS_DIR / f'label_encoder_{col}.pkl'
    try:
        LOADED_LABEL_ENCODERS[col] = joblib.load(encoder_path)
        print(f"✅ LabelEncoder para '{col}' cargado correctamente.")
        # La línea de depuración solo para el inicio del servidor, luego se puede eliminar
        # print(f"DEBUG: LabelEncoder '{col}' en servidor tiene clases: {LOADED_LABEL_ENCODERS[col].classes_}", file=sys.stderr)
    except FileNotFoundError:
        print(f"❌ Error: Archivo de LabelEncoder para '{col}' no encontrado en {encoder_path}.", file=sys.stderr)
        print("Asegúrate de haber guardado todos los LabelEncoders individuales en Colab y colocado en la carpeta 'models'.", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"❌ Error cargando LabelEncoder para '{col}': {str(e)}", file=sys.stderr)
        sys.exit(1)

# --- Variables para Imputación y Mapeo (ACTUALIZA ESTOS VALORES CON LOS REALES DE TU ENTRENAMIENTO) ---
DEFAULT_AGE_VALUE = 30 # <-- ¡ACTUALIZA! (ej. media/mediana de 'Age' de tu dataset de entrenamiento)
DEFAULT_WORK_INTERFERE_VALUE = "Never" # <-- ¡ACTUALIZA! (ej. moda de 'work_interfere' de tu dataset de entrenamiento)

# Mapeos explícitos para estandarizar inputs a lo que el LabelEncoder REALMENTE aprendió.
# Las claves son los valores que esperas del formulario (Male, Female, Never, etc.)
# Los VALORES son las categorías EXACTAS que el LabelEncoder aprendió en Colab.
GENDER_MAPPING = {
    'Male': 'Male',         
    'Female': 'Female',     
    'Other': 'Other',       
    'M': 'Male',            # Mapea 'M' a 'Male' si el encoder aprendió 'Male'
    'F': 'Female',          # Mapea 'F' a 'Female' si el encoder aprendió 'Female'
    'male': 'Male',         # Mapea 'male' a 'Male'
    'female': 'Female',     # Mapea 'female' a 'Female'
    'other': 'Other',       # Mapea 'other' a 'Other'
    # Agrega aquí CUALQUIER OTRA VARIACIÓN que tu LabelEncoder haya aprendido
    # y que quieras mapear a 'Male', 'Female' u 'Other'.
    # Ejemplo: 'cis male': 'Male', 'Trans woman': 'Female', 'Agender': 'Other'
    'DEFAULT': 'Other' # Valor por defecto si la entrada no coincide con ninguna clave
}

WORK_INTERFERE_MAPPING = {
    'Never': 'Never',
    'Rarely': 'Rarely',
    'Sometimes': 'Sometimes',
    'Often': 'Often',
    # Si tu dataset original tenía 'NA' o 'nan' string y el encoder los aprendió:
    # 'NA': 'NA',
    'DEFAULT': DEFAULT_WORK_INTERFERE_VALUE
}

# Diccionario de valores por defecto para columnas categóricas que puedan faltar o ser "unseen"
# Los valores aquí DEBEN ser las categorías EXACTAS (con sus mayúsculas/minúsculas correctas)
# que el LabelEncoder para esa columna aprendió.
default_categorical_values = {
    'Gender': GENDER_MAPPING['DEFAULT'],
    'Country': 'United States', # O la categoría de país más común en tu entrenamiento
    'self_employed': 'No',
    'family_history': 'No',
    'treatment': 'No',
    'no_employees': '1-5',
    'remote_work': 'No',
    'tech_company': 'No',
    'benefits': "Don't know",
    'care_options': 'Not sure',
    'wellness_program': "Don't know",
    'seek_help': "Don't know",
    'anonymity': "Don't know",
    'leave': "Don't know",
    'mental_health_consequence': 'Maybe',
    'phys_health_consequence': 'Maybe',
    'coworkers': 'Some of them',
    'supervisor': 'Some of them',
    'mental_health_interview': 'Maybe',
    'phys_health_interview': 'Maybe',
    'mental_vs_physical': "Don't know",
    'obs_consequence': 'No',
    'work_interfere': WORK_INTERFERE_MAPPING['DEFAULT']
}


# --- Función de Preprocesamiento ---
def preprocess_input(form_data):
    """
    Preprocesa los datos de entrada del formulario para que el modelo pueda hacer una predicción.
    Asegura que todas las características esperadas estén presentes y en el orden correcto.
    """
    input_data = pd.DataFrame([form_data])

    if 'Age' in input_data.columns:
        input_data['Age'] = pd.to_numeric(input_data['Age'], errors='coerce')
        input_data['Age'] = input_data['Age'].fillna(DEFAULT_AGE_VALUE)
    else:
        input_data['Age'] = DEFAULT_AGE_VALUE
        print("Advertencia: 'Age' no se encontró en los datos del formulario, usando valor por defecto.", file=sys.stderr)

    for col in CATEGORICAL_COLS_TO_ENCODE:
        if col in input_data.columns:
            le = LOADED_LABEL_ENCODERS[col]
            
            input_value = str(input_data[col].iloc[0]).strip()

            input_value_to_transform = input_value 
            if col == 'Gender':
                # Aplica el mapeo de Gender (case-insensitive para las claves del mapeo, pero el valor resultante debe ser exacto)
                input_value_to_transform = GENDER_MAPPING.get(input_value, GENDER_MAPPING['DEFAULT'])
                print(f"DEBUG: Gender input '{input_value}' mapeado a '{input_value_to_transform}'.", file=sys.stderr)
            elif col == 'work_interfere':
                input_value_to_transform = WORK_INTERFERE_MAPPING.get(input_value, WORK_INTERFERE_MAPPING['DEFAULT'])
                print(f"DEBUG: work_interfere input '{input_value}' mapeado a '{input_value_to_transform}'.", file=sys.stderr)
            # Para otras columnas, se utiliza el valor de entrada directamente después de strip()
            # Si estas también tienen inconsistencias de mayúsculas/minúsculas o sinónimos,
            # DEBERÍAS AÑADIR MAPEO ESPECÍFICO para ellas también, similar a Gender.

            # Manejo de valores vacíos o 'Seleccione...'
            if input_value_to_transform in ['', 'nan', 'na', 'Seleccione...']:
                if col in default_categorical_values:
                    input_value_to_transform = default_categorical_values[col]
                    print(f"DEBUG: '{col}' era vacío/inválido, re-imputado con '{input_value_to_transform}'.", file=sys.stderr)
                else:
                    raise ValueError(f"Valor no válido o faltante para la pregunta '{col}'. No hay un valor por defecto definido. Por favor, verifica tu selección.")

            try:
                input_data[col] = le.transform([input_value_to_transform])[0]
            except ValueError as ve:
                print(f"Error: El valor '{input_value_to_transform}' no es reconocido para la columna '{col}'. Detalle: {ve}", file=sys.stderr)
                raise ValueError(f"Valor no válido para la pregunta '{col}': '{input_value}'. El valor final '{input_value_to_transform}' no es una categoría conocida por el modelo. Verifica tu mapeo o las clases del LabelEncoder.") from ve
        else:
            print(f"Advertencia: Columna categórica '{col}' esperada por el preprocesador no encontrada en los datos del formulario. Imputando valor por defecto.", file=sys.stderr)
            if col in default_categorical_values:
                imputed_value = default_categorical_values[col]
                input_data[col] = LOADED_LABEL_ENCODERS[col].transform([imputed_value])[0]
                print(f"DEBUG: '{col}' faltaba en el formulario, imputado con '{imputed_value}'.", file=sys.stderr)
            else:
                raise ValueError(f"Falta una pregunta necesaria en el formulario: '{col}' y no se ha definido un valor por defecto para ella. Por favor, complete todas las secciones.")

    final_processed_data = pd.DataFrame(index=[0], columns=GLOBAL_MODEL_FEATURES)

    for col in GLOBAL_MODEL_FEATURES:
        if col in input_data.columns:
            final_processed_data[col] = input_data[col].iloc[0]
        else:
            print(f"Error CRÍTICO: La columna esperada por el modelo '{col}' no se encuentra en los datos procesados del usuario. Esto es crítico.", file=sys.stderr)
            raise ValueError(f"Falta una característica crítica para la predicción: '{col}'. Verifique que todas las preguntas del formulario coincidan con las características del modelo y que el preprocesamiento las genere correctamente.")

    final_processed_data = final_processed_data[GLOBAL_MODEL_FEATURES]
    final_processed_data = final_processed_data.apply(pd.to_numeric, errors='raise')
    
    return final_processed_data