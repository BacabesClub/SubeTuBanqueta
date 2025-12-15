# app/routes.py
import os
import uuid
from datetime import datetime
from flask import (
    Blueprint, render_template, request, jsonify, current_app, 
    url_for, redirect,
    send_file, make_response, current_app
)
import io
import zipfile
import csv

from werkzeug.utils import secure_filename
# MODIFICADO: Importamos Report en lugar de Image
from app.models import Report
from app.extensions import db
from app.services import classify_image_local

main = Blueprint('main', __name__)

@main.route('/')
def index():
    return render_template('index.html')

@main.route('/mapa')
def mapa():
    return render_template('mapa.html')

@main.route('/api/points')
def api_points():
    # MODIFICADO: Usamos Report
    points_from_db = Report.query.all()
    points = [p.to_dict() for p in points_from_db]
    return jsonify(points)

# Esta es la ruta original, ahora para 'accesibilidad'
@main.route('/upload', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        file = request.files.get('file')
        if not file or not file.filename:
            return "No se subió ningún archivo", 400

        # ... (lógica de file_name, filepath, etc. se mantiene igual)
        extension = os.path.splitext(file.filename)[1]
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        unique_id = str(uuid.uuid4().hex)[:8]
        filename = secure_filename(f"{timestamp}_{unique_id}{extension}")
        filepath = os.path.join(current_app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath) 
        
        user_accessibility_level = request.form.get('accessibility', 'No especificado')
        observations_text = request.form.get('observations', '')
        lat = request.form.get('latitude')
        lon = request.form.get('longitude')

        if not lat or not lon:
            return "Error: No se recibió la ubicación.", 400
        
        location = f"{lat},{lon}"
        
        try:
            print(f"Ejecutando predicción local para {filename}...")
            model_prediction = classify_image_local(filepath)
            print(f"Predicción del modelo: {model_prediction}")

            # MODIFICADO: Usamos Report y añadimos el tipo
            new_report = Report(
                file_name=filename,
                location=location,
                observations=observations_text,
                report_type='accessibility', # <-- TIPO DE REPORTE
                user_accessibility=user_accessibility_level,
                model_accessibility=model_prediction
            )

            db.session.add(new_report)
            db.session.commit()
            new_id = new_report.id

            # MODIFICADO: Pasamos el tipo y un 'back_url' al resultado
            return render_template('result.html',
                                   filename=filename,
                                   accessibility=user_accessibility_level,
                                   model_prediction=model_prediction,
                                   new_id=new_id,
                                   report_type='accessibility',
                                   back_url=url_for('main.upload')) # <-- Link para 'Subir otra'

        except Exception as e:
            db.session.rollback()
            print(f"Error al guardar en la BD o predecir: {e}")
            return "Error al guardar el registro.", 500

    # El GET sigue igual
    return render_template('upload.html')


# --- ¡NUEVA RUTA PARA OBSTÁCULOS! ---
@main.route('/reportar_obstaculo', methods=['GET', 'POST'])
def reportar_obstaculo():
    if request.method == 'POST':
        file = request.files.get('file')
        if not file or not file.filename:
            return "No se subió ningún archivo", 400

        # Lógica de guardado de archivo (igual que en upload)
        extension = os.path.splitext(file.filename)[1]
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        unique_id = str(uuid.uuid4().hex)[:8]
        filename = secure_filename(f"{timestamp}_{unique_id}{extension}")
        filepath = os.path.join(current_app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath) 
        
        # SOLO pedimos observaciones y ubicación
        observations_text = request.form.get('observations', '')
        lat = request.form.get('latitude')
        lon = request.form.get('longitude')

        if not lat or not lon:
            return "Error: No se recibió la ubicación.", 400
        
        location = f"{lat},{lon}"
        
        try:
            # NO hay predicción del modelo aquí

            # MODIFICADO: Usamos Report y el nuevo tipo
            new_report = Report(
                file_name=filename,
                location=location,
                observations=observations_text,
                report_type='obstacle' # <-- TIPO DE REPORTE
                # user_accessibility y model_accessibility quedan NULL
            )

            db.session.add(new_report)
            db.session.commit()
            new_id = new_report.id

            # Pasamos los datos al mismo template 'result.html'
            return render_template('result.html',
                                   filename=filename,
                                   observations=observations_text,
                                   new_id=new_id,
                                   report_type='obstacle', # <-- TIPO DE REPORTE
                                   back_url=url_for('main.reportar_obstaculo')) # <-- Link para 'Subir otra'

        except Exception as e:
            db.session.rollback()
            print(f"Error al guardar en la BD: {e}")
            return "Error al guardar el registro.", 500

    # Para el GET, renderizamos un nuevo template que es copia de 'upload.html'
    return render_template('reportar_obstaculo.html')


@main.route('/database')
def database():
    # MODIFICADO: Usamos Report
    rows = Report.query.order_by(Report.id.desc()).all()
    return render_template('database.html', rows=rows)

@main.route('/admin/export-data')
def export_data_csv():
    """
    Descarga la base de datos actual como CSV.
    Fuente de verdad: SQL (Report).
    """
    # 1. Consultar la Fuente de Verdad
    reports = Report.query.all()

    # 2. Crear el CSV en memoria (RAM)
    si = io.StringIO()
    writer = csv.writer(si)

    # Escribir cabeceras (Mapea esto a lo que necesita tu modelo de IA para re-entrenar)
    writer.writerow(['id', 'imagen_nombre', 'lat', 'lon', 'observaciones', 'etiqueta_real', 'prediccion', 'fecha'])

    # 3. Llenar datos
    for r in reports:
        # Separar lat/lon si vienen juntos
        parts = r.location.split(',')
        lat = parts[0] if len(parts) > 0 else ''
        lon = parts[1] if len(parts) > 1 else ''

        writer.writerow([
            r.id,
            r.file_name,
            lat,
            lon,
            r.observations,
            r.user_accessibility,   # Esta es la etiqueta valiosa (corrección humana)
            r.model_accessibility,  # Esta es la referencia de la IA
            r.date.isoformat()
        ])

    # GENERAR NOMBRE CON FECHA ISO (Ej: dataset_2025-12-14_1730.csv)
    timestamp = datetime.now().strftime("%Y-%m-%d_%H%M")
    filename = f"dataset_snapshot_{timestamp}.csv"

    output = make_response(si.getvalue())
    output.headers["Content-Disposition"] = f"attachment; filename={filename}" # <--- Nombre dinámico
    output.headers["Content-type"] = "text/csv"
    return output

@main.route('/admin/export-images')
def export_images_zip():
    """
    Descarga un ZIP con SOLO las imágenes que existen en la BD.
    Limpia automáticamente la basura (imágenes huérfanas no se descargan).
    """
    # 1. Consultar la Fuente de Verdad
    reports = Report.query.all()
    upload_folder = current_app.config['UPLOAD_FOLDER']

    # 2. Crear buffer en memoria para el ZIP
    memory_file = io.BytesIO()

    # 3. Crear el ZIP
    # 'w' mode escribe, zipfile.ZIP_DEFLATED comprime para ahorrar espacio
    with zipfile.ZipFile(memory_file, 'w', zipfile.ZIP_DEFLATED) as zf:
        added_count = 0
        for r in reports:
            filename = r.file_name
            file_path = os.path.join(upload_folder, filename)

            # Solo agregamos si el archivo REALMENTE existe en disco
            if os.path.exists(file_path):
                # arcname es el nombre que tendrá DENTRO del zip (sin rutas raras)
                zf.write(file_path, arcname=filename)
                added_count += 1

    # 4. Rebobinar el puntero del archivo al inicio para leerlo
    memory_file.seek(0)

    print(f"📦 ZIP generado con {added_count} imágenes válidas.")
    # GENERAR NOMBRE CON FECHA ISO
    timestamp = datetime.now().strftime("%Y-%m-%d_%H%M")
    filename = f"imagenes_snapshot_{timestamp}.zip"

    # ...
    return send_file(
        memory_file,
        mimetype='application/zip',
        as_attachment=True,
        download_name=filename # <--- Nombre dinámico
    )
