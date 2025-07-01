import os
import sqlite3
import random
from flask import Flask, render_template, request, redirect, url_for, jsonify
from werkzeug.utils import secure_filename
import google.generativeai as genai
from PIL import Image

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'

# --- CONFIGURACIÓN DE LA API DE GEMINI ---
try:
    genai.configure(api_key="AIzaSyC6rA2b3mDnN6BncdQA9Ah_RuKpE6n9V0A")
except KeyError:
    try:
        # Pega tu clave aquí si no usas variables de entorno
        
        genai.configure(api_key=os.environ["GOOGLE_API_KEY"])
    except Exception as e:
        print("Error: Clave de API no configurada.")

def classify_image(image_path):
    """
    Envía una imagen a la API de Gemini Vision para clasificar su nivel de accesibilidad.
    """
    try:
        img_to_classify = Image.open(image_path)
        model = genai.GenerativeModel('gemini-1.5-flash')
        
        prompt_parts = [
            """
            Tarea: Analiza la imagen de una vía pública (acera, calle, etc.) y clasifícala estrictamente en uno de los siguientes cuatro niveles de accesibilidad.

            Niveles de Accesibilidad:
            Accesible: La superficie presenta grietas e irregularidades de baja altura. El desplazamiento es posible con poca o mediana dificultad. Riesgos: Multiples grietas, fracturas ramificada, desniveles leves. Dificultad: Baja/Media.
            Medianamente accesible: La superficie tiene irregularidades con un desnivel visible del plano original pero transitable. adoquines levantados. Aunque es posible transitar, requiere un esfuerzo moderado. Riesgos: Levantamientos pequeños, pero que no llegan a escalon, separacion amplia entre losas. Dificultad: Media/Alta. 
            Poco accesible: La superficie tiene irregularidades con un desnivel no transitable, obstáculos significativos como raíces expuestas, hoyos marcados, o rupturas en la banqueta. El desplazamiento requiere maniobras evasivas constantes. Riesgos:     Fracturas estructurales profundas, Desniveles pronunciados y descalces abruptos, Huecos de socavación expuestos (o Cavidades internas visibles), Raices expuestas, tramos faltantes amplios en ancho. Dificultad: Extrema.
            Inaccesible: La superficie contiene obstáculos intransitables como coladeras (abiertas o cerradas), bordes de banqueta, escalones, o rampas con pendientes excesivas. Riesgos: coladeras, bordes de banqueta, escalones, rampas. Dificultad: Imposible.

            Instrucciones de Respuesta:
            1.  Tu respuesta DEBE SER ÚNICAMENTE una de las cuatro etiquetas de nivel definidas arriba (Accesible, Medianamente accesible, Poco accesible, Inaccesible).
            2.  No incluyas explicaciones, números, ni ningún otro texto.
            3.  Si la imagen es de muy baja calidad, está oscura o no es relevante, clasifícala como "Inaccesible".

            Ejemplo de respuesta válida:
            Poco accesible

            Analiza la siguiente imagen y proporciona tu clasificación.
            """,
            img_to_classify,
        ]

        response = model.generate_content(prompt_parts)
        # Limpiar la respuesta para que coincida con los niveles esperados
        cleaned_response = response.text.strip().replace("'", "").replace("`", "")
        
        valid_levels = ["Accesible", "Medianamente accesible", "Poco accesible", "Inaccesible"]
        
        # Forzar la respuesta a uno de los niveles válidos
        final_level = "Inaccesible" # Valor por defecto en caso de respuesta inesperada
        for level in valid_levels:
            if level.lower() in cleaned_response.lower():
                final_level = level
                break
        
        return final_level

    except Exception as e:
        print(f"Error en la clasificación de imagen: {e}")
        return "Inaccesible"

def init_db():
    conn = sqlite3.connect('database.db')
    print("Opened database successfully")
    conn.execute('CREATE TABLE IF NOT EXISTS images (id INTEGER PRIMARY KEY AUTOINCREMENT, file_name TEXT, location TEXT, accessibility TEXT, date TEXT)')
    print("Table created successfully")
    conn.close()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/mapa')
def mapa():
    return render_template('mapa.html')

@app.route('/api/points')
def api_points():
    conn = sqlite3.connect('database.db')
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    cursor.execute("SELECT file_name, location, accessibility FROM images")
    rows = cursor.fetchall()
    conn.close()
    points = [dict(row) for row in rows]
    return jsonify(points)


@app.route('/upload', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        file = request.files['file']
        if file:
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            
            # Clasificar la imagen
            accessibility_level = classify_image(filepath)

            # Generar coordenadas aleatorias en CDMX
            lat = random.uniform(19.2, 19.6)
            lon = random.uniform(-99.3, -98.9)
            location = f"{lat},{lon}"
            
            conn = sqlite3.connect('database.db')
            cursor = conn.cursor()
            cursor.execute("INSERT INTO images (file_name, location, accessibility, date) VALUES (?, ?, ?, datetime('now'))", 
                           (filename, location, accessibility_level))
            conn.commit()
            conn.close()
            
            return render_template('result.html', filename=filename, accessibility=accessibility_level)
    return render_template('upload.html')

@app.route('/database')
def database():
    conn = sqlite3.connect('database.db')
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM images")
    rows = cursor.fetchall()
    conn.close()
    return render_template('database.html', rows=rows)

if __name__ == '__main__':
    if not os.path.exists(app.config['UPLOAD_FOLDER']):
        os.makedirs(app.config['UPLOAD_FOLDER'])
    init_db()
    app.run(host='0.0.0.0', port=5000, debug=True)
