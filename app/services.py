# /home/admin54/SubeTuBanqueta/app/services.py

import os
import csv
import zipfile
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image as PILImage
import google.generativeai as genai

# --- IMPORTACIÓN ESTRELLA PARA HUGGING FACE ---
from huggingface_hub import hf_hub_download

# Importaciones de tu app
from app.extensions import db
from app.models import Report
from datetime import datetime
# --- CONFIGURACIÓN DE LOS REPOSITORIOS (LA FUENTE DE VERDAD) ---
MODEL_REPO_ID = "joshieadalid/SubeTuBanqueta-Modelo"
DATA_REPO_ID = "joshieadalid/SubeTuBanqueta"  # Repo para datos

MODEL_FILENAME = "modelo_banquetas.pt"
CSV_FILENAME = "data.csv"       # CSV con metadatos
ZIP_FILENAME = "images.zip"     # ZIP con imágenes

# --- VARIABLES GLOBALES ---
LOCAL_MODEL = None
CLASS_NAMES = ['Accesible', 'Medianamente accesible', 'Poco accesible', 'Inaccesible']
DEVICE = None
VAL_TRANSFORMS = None

# --- 1. TRANSFORMACIONES ---
def get_val_transforms(img_size=224):
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

# --- 2. INICIALIZADOR DEL MODELO ---
def init_local_model(app):
    """Descarga el modelo de Hugging Face y lo carga en memoria."""
    global LOCAL_MODEL, DEVICE, VAL_TRANSFORMS
    
    print(f">> [HuggingFace] Verificando modelo '{MODEL_FILENAME}' en '{MODEL_REPO_ID}'...")

    try:
        DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        VAL_TRANSFORMS = get_val_transforms()

        # Descarga automática desde Hugging Face
        model_path = hf_hub_download(repo_id=MODEL_REPO_ID, filename=MODEL_FILENAME)
        print(f"✅ Modelo asegurado en ruta local: {model_path}")

        # Construir arquitectura (ConvNeXt Base)
        try:
            weights = models.ConvNeXt_Base_Weights.DEFAULT
            LOCAL_MODEL = models.convnext_base(weights=weights)
        except:
            LOCAL_MODEL = models.convnext_base(pretrained=True)

        # Ajustar capa final para 4 clases
        num_ftrs = LOCAL_MODEL.classifier[2].in_features
        LOCAL_MODEL.classifier[2] = nn.Linear(num_ftrs, 4)

        # Cargar pesos entrenados
        checkpoint = torch.load(model_path, map_location=DEVICE, weights_only=False)
        
        if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
            LOCAL_MODEL.load_state_dict(checkpoint['state_dict'])
        elif isinstance(checkpoint, dict):
            LOCAL_MODEL.load_state_dict(checkpoint)
        else:
            LOCAL_MODEL = checkpoint

        LOCAL_MODEL.to(DEVICE)
        LOCAL_MODEL.eval()
        
        print("="*50)
        print(f"✅ MODELO CARGADO EXITOSAMENTE DESDE HUGGING FACE")
        print(f"   Clases: {CLASS_NAMES}")
        print(f"   Dispositivo: {DEVICE}")
        print("="*50)

    except Exception as e:
        print(f"❌ ERROR FATAL AL CARGAR EL MODELO: {e}")
        LOCAL_MODEL = None

# --- 3. CLASIFICADOR LOCAL ---
def classify_image_local(image_path):
    """Clasifica una imagen usando el modelo cargado en memoria."""
    global LOCAL_MODEL

    if LOCAL_MODEL is None:
        print("⚠️ Intento de clasificación pero el modelo no está cargado.")
        return "Error (Modelo no cargado)"
    
    try:
        image = PILImage.open(image_path).convert("RGB")
        input_tensor = VAL_TRANSFORMS(image)
        input_batch = input_tensor.unsqueeze(0).to(DEVICE)
        
        with torch.no_grad():
            logits = LOCAL_MODEL(input_batch)
            
        pred_idx = logits.argmax(dim=1).item()
        return CLASS_NAMES[pred_idx]
    
    except Exception as e:
        print(f"Error durante la inferencia local: {e}")
        return "Error de Inferencia"

# --- 4. HIDRATACIÓN DE DATOS ---
def seed_database(app):
    """
    AUTO-HIDRATACIÓN IDEMPOTENTE:
    Descarga datos de Hugging Face y puebla la BD sin crear duplicados.
    """
    print(">> [Data Bootstrap] Verificando estado de la memoria...")

    # NOTA: Ya no hacemos el "return" si hay datos, porque queremos
    # poder correr esto para agregar datos nuevos sin borrar los viejos.
    
    upload_folder = app.config['UPLOAD_FOLDER']
    os.makedirs(upload_folder, exist_ok=True)

    try:
        # 1. Descargar recursos si no existen (Caché inteligente de HF)
        print(f"⬇️ Sincronizando imágenes: {ZIP_FILENAME}...")
        zip_path = hf_hub_download(
            repo_id=DATA_REPO_ID, 
            filename=ZIP_FILENAME, 
            repo_type="dataset"
        )
        
        # Descomprimir siempre es seguro (sobreescribe archivos si ya existen)
        print(f"📦 Descomprimiendo en {upload_folder}...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(upload_folder)

        print(f"⬇️ Sincronizando datos: {CSV_FILENAME}...")
        csv_path = hf_hub_download(
            repo_id=DATA_REPO_ID, 
            filename=CSV_FILENAME, 
            repo_type="dataset"
        )
        
        print("📄 Procesando CSV e importando a SQL...")
        row_count = 0
        
        # --- OPTIMIZACIÓN: Cargar catálogo existente en RAM ---
        # Esto evita hacer miles de consultas a la BD.
        # Creamos un conjunto (set) con los nombres de archivo que YA tenemos.
        existing_filenames = {
            r[0] for r in db.session.query(Report.file_name).all()
        }
        
        with open(csv_path, newline='', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            
            for row in reader:
                filename = row.get('imagen_nombre')
                
                # A. FILTRO DE DUPLICADOS (La Vacuna 💉)
                # Si el archivo ya está en la BD, saltamos al siguiente.
                if filename in existing_filenames:
                    continue

                # B. FILTRO DE BASURA (Coordenadas)
                lat_raw = row.get('lat')
                lon_raw = row.get('lon')
                
                # Si no hay latitud o dice 'unknown', es basura.
                if not lat_raw or not lon_raw or 'unknown' in str(lat_raw).lower():
                    continue

                # C. PARSEO DE FECHA HISTÓRICA
                # Intentamos leer la fecha del CSV. Si falla, usamos 'ahora'.
                fecha_csv = row.get('fecha')
                report_date = datetime.now()
                
                if fecha_csv:
                    try:
                        # Intenta formato ISO (2025-12-14T12:00:00) o estándar
                        report_date = datetime.fromisoformat(str(fecha_csv).replace('Z', ''))
                    except ValueError:
                        pass # Si falla, se queda con datetime.now()

                # D. CREACIÓN DEL REPORTE
                new_report = Report(
                    file_name = filename, 
                    location = f"{lat_raw},{lon_raw}",
                    observations = row.get('observaciones', ''),
                    report_type = 'dataset_historico',
                    user_accessibility = row.get('etiqueta_real'), # Tu etiqueta manual
                    model_accessibility = row.get('prediccion'),   # La predicción guardada
                    date = report_date # Usamos la fecha real del evento
                )
                
                db.session.add(new_report)
                
                # Agregamos al set local para evitar duplicados dentro del mismo CSV
                existing_filenames.add(filename)
                row_count += 1
        
        db.session.commit()
        
        if row_count > 0:
            print(f"✅ ¡Hidratación completada! Se insertaron {row_count} registros NUEVOS.")
        else:
            print("✅ El sistema está actualizado. No se encontraron datos nuevos.")

    except Exception as e:
        db.session.rollback()
        print(f"❌ Error en Hidratación: {e}")

# --- 5. FUNCIONES GEMINI (Respaldo) ---
def configure_genai(api_key):
    try:
        genai.configure(api_key=api_key)
    except Exception as e:
        print(f"Error config Gemini: {e}")

def classify_image(image_path):
    """Clasificación con Gemini (Respaldo)"""
    try:
        img_to_classify = PILImage.open(image_path)
        model = genai.GenerativeModel('gemini-1.5-flash')
        prompt = "Analiza la accesibilidad de esta banqueta. Responde solo con una de estas opciones: Accesible, Medianamente accesible, Poco accesible, Inaccesible."
        
        response = model.generate_content([prompt, img_to_classify])
        text = response.text.strip()
        
        # Limpieza básica
        for nivel in CLASS_NAMES:
            if nivel.lower() in text.lower():
                return nivel
        return "Inaccesible" # Default pesimista
    except Exception as e:
        print(f"Error Gemini: {e}")
        return "Error (Gemini)"
