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
    AUTO-HIDRATACIÓN:
    Si la base de datos está vacía, descarga datos desde Hugging Face Datasets,
    restaura las imágenes y registros históricos.
    """
    print(">> [Data Bootstrap] Verificando estado de la memoria...")

    # Verificar si ya hay datos
    if Report.query.first() is not None:
        print("✅ Base de datos ya poblada. Sistema listo para operar.")
        return

    print("💧 Sistema vacío detectado. Iniciando protocolo de hidratación desde Hugging Face...")
    
    upload_folder = app.config['UPLOAD_FOLDER']
    os.makedirs(upload_folder, exist_ok=True)

    try:
        # Descargar y descomprimir imágenes
        print(f"⬇️ Descargando imágenes: {ZIP_FILENAME}...")
        zip_path = hf_hub_download(
            repo_id=DATA_REPO_ID, 
            filename=ZIP_FILENAME, 
            repo_type="dataset"
        )
        
        print(f"📦 Descomprimiendo en {upload_folder}...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(upload_folder)
        print("✅ Imágenes restauradas.")

        # Descargar y procesar datos
        print(f"⬇️ Descargando datos: {CSV_FILENAME}...")
        csv_path = hf_hub_download(
            repo_id=DATA_REPO_ID, 
            filename=CSV_FILENAME, 
            repo_type="dataset"
        )
        
        print("📄 Importando registros a SQL...")
        row_count = 0
        
        with open(csv_path, newline='', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            
            for row in reader:
                new_report = Report(
                    file_name = row.get('imagen_nombre'), 
                    location = f"{row.get('lat')},{row.get('lon')}",
                    observations = row.get('observaciones', ''),
                    report_type = 'dataset_historico',
                    user_accessibility = row.get('etiqueta_real'),
                    model_accessibility = row.get('prediccion'),
                )
                db.session.add(new_report)
                row_count += 1
        
        db.session.commit()
        print(f"✅ ¡Hidratación completada! Se restauraron {row_count} registros.")

    except Exception as e:
        db.session.rollback()
        print(f"❌ Error crítico en Hidratación: {e}")
        print("   La app iniciará vacía (sin datos históricos).")

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
