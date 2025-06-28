import sys
import os
from pathlib import Path

# Importaciones de Flask y otras librerías necesarias
from flask import Flask, render_template, request, jsonify
import joblib
import pandas as pd
import numpy as np # <-- Ahora que las versiones coinciden, no necesitamos el parche de np.core
from preprocessing import preprocess_input # Importa tu función de preprocesamiento

# --- Configuración y Carga del Modelo ---
# Configuración de rutas seguras
MODEL_DIR = Path('models')
MODEL_PATH = MODEL_DIR / 'modelo_random_forest_salud_mental.pkl'

# Verificación de archivos
if not MODEL_PATH.exists():
    # Es mejor que la aplicación falle al inicio si el modelo no está,
    # en lugar de intentar cargar algo que no existe.
    raise FileNotFoundError(f"Error: Modelo no encontrado en {MODEL_PATH}. Asegúrate de que el archivo .pkl esté en la carpeta 'models'.")

# Cargar el modelo entrenado
# La carga del modelo se realiza una sola vez al iniciar la aplicación.
# Si la carga falla aquí, la aplicación no se iniciará, lo cual es deseable.
try:
    model = joblib.load(MODEL_PATH)
    print("✅ Modelo cargado correctamente al inicio de la aplicación.")
except Exception as e:
    print(f"❌ Error crítico al cargar el modelo: {str(e)}", file=sys.stderr)
    # Aquí podrías decidir si la aplicación debe salir o no,
    # pero para una app web que depende del modelo, es mejor salir.
    sys.exit(1) # Salir de la aplicación si el modelo no puede cargar


# --- Inicialización de la Aplicación Flask ---
app = Flask(__name__)

# --- Rutas de la Aplicación ---

@app.route('/')
def home():
    """
    Ruta principal que renderiza el formulario de la encuesta.
    """
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    """
    Ruta para manejar las predicciones del modelo.
    Recibe los datos del formulario, los preprocesa y devuelve el resultado.
    """
    try:
        # Obtener datos del formulario
        form_data = request.form.to_dict()
        
        # Preprocesar los datos de entrada
        # Asume que preprocess_input ya maneja los LabelEncoders y el orden de las columnas.
        processed_data = preprocess_input(form_data)
        
        # Asegúrate de que 'processed_data' sea un DataFrame con las columnas correctas
        # y en el orden esperado por el modelo.
        # Si preprocess_input devuelve un diccionario o Series, conviértelo a DataFrame.
        # Por ejemplo: processed_data = pd.DataFrame([processed_data]) si devuelve un dict.

        # Realizar la predicción
        prediction = model.predict(processed_data)
        probability = model.predict_proba(processed_data)[0]
        
        # Interpretar resultados
        # Asegúrate de que 'prediction' y 'probability' tengan los valores esperados.
        # Por ejemplo, si tu modelo predice 0 para 'Sin preocupación' y 1 para 'Preocupación'.
        result = "Preocupación por salud mental" if prediction[0] == 1 else "Sin preocupación significativa"
        
        # La confianza es la probabilidad de la clase predicha.
        # Si prediction[0] es 1 (Preocupación), confidence es probability[1]
        # Si prediction[0] es 0 (Sin preocupación), confidence es probability[0]
        confidence = probability[prediction[0]] # Probabilidad de la clase predicha
        
        return render_template('results.html', 
                             result=result, 
                             confidence=f"{confidence*100:.2f}%")
    
    except Exception as e:
        # En un entorno de producción, loggea el error pero no lo muestres directamente al usuario.
        # Para depuración, es útil mostrarlo.
        print(f"Error en la ruta /predict: {str(e)}", file=sys.stderr)
        return render_template('results.html', 
                             error=f"Lo sentimos, ha ocurrido un error al procesar tu solicitud: {str(e)}")

# --- Ejecución de la Aplicación ---
if __name__ == '__main__':
    # Usar debug=True solo durante el desarrollo.
    # Para producción, desactívalo y usa un servidor WSGI como Gunicorn.
    app.run(debug=True)