# /home/admin54/SubeTuBanqueta/config.py

import os

basedir = os.path.abspath(os.path.dirname(__file__))

class Config:
    """Clase de configuración base."""
    SECRET_KEY = os.environ.get('SECRET_KEY') or 'una-clave-secreta-muy-dificil'
    
    SQLALCHEMY_DATABASE_URI = os.environ.get('DATABASE_URL') or \
        'sqlite:///' + os.path.join(basedir, 'database.db')
    SQLALCHEMY_TRACK_MODIFICATIONS = False
    
    UPLOAD_FOLDER = os.path.join(basedir, 'app/static/uploads')
    
    GOOGLE_API_KEY = os.environ.get('GOOGLE_API_KEY')
    
    # --- AÑADE ESTA LÍNEA ---
    # Ruta a tu modelo PyTorch entrenado
    LOCAL_MODEL_PATH = os.path.join(basedir, 'modelo/best_model_convnext_base_ordinal_dataset2_csv.pt')
