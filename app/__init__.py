# /home/admin54/SubeTuBanqueta/app/__init__.py

import os
from flask import Flask
from werkzeug.middleware.proxy_fix import ProxyFix
import click
from flask.cli import with_appcontext

from config import Config
from app.extensions import db
from app.services import configure_genai, init_local_model, seed_database
def create_app(config_class=Config):
    """La Función Factory de la Aplicación."""
    
    app = Flask(__name__) 
    app.config.from_object(config_class)
    app.wsgi_app = ProxyFix(app.wsgi_app, x_for=1, x_proto=1, x_host=1, x_prefix=1)

    upload_folder = app.config['UPLOAD_FOLDER']
    if not os.path.exists(upload_folder):
        os.makedirs(upload_folder, exist_ok=True)
        print(f">> [System] Carpeta de uploads asegurada: {upload_folder}")

    # Inicializa la Base de Datos
    db.init_app(app)

    with app.app_context():
        # IMPORTANTE: Importar el modelo aquí para que SQLAlchemy sepa qué crear
        from app.models import Report 
        db.create_all()
        print(">> Sistema de Base de Datos verificado/inicializado.")
        
        # --- NUEVO: AQUÍ LLAMAMOS A LA HIDRATACIÓN ---
        # Esto revisará si la BD está vacía y bajará los datos de Hugging Face
        try:
            seed_database(app)
        except Exception as e:
            print(f"❌ Error en el proceso de Seed/Hidratación: {e}")
        # ---------------------------------------------
    # Configura la API de Gemini (la dejamos por si la quieres de respaldo)
    if app.config['GOOGLE_API_KEY']:
        configure_genai(app.config['GOOGLE_API_KEY'])
    else:
        print("ADVERTENCIA: GOOGLE_API_KEY no está configurada.")
        
    # --- AÑADE ESTO: Carga el modelo local PyTorch ---
    # Lo hacemos dentro de un try/except para que la app
    # no falle si torch no está instalado.
    try:
        init_local_model(app)
    except ImportError:
        print("="*50)
        print("ADVERTENCIA: 'torch' o 'torchvision' no están instalados.")
        print("El modelo local PyTorch no se cargará.")
        print("Instálalos con: pip install torch torchvision")
        print("="*50)
    except Exception as e:
        print(f"ADVERTENCIA: No se pudo cargar el modelo local PyTorch.")
        print(f"Error: {e}")
        print("="*50)
    # --- FIN DE LA SECCIÓN AÑADIDA ---
        
    # Registra los Blueprints (rutas)
    from app.routes import main as main_blueprint
    app.register_blueprint(main_blueprint)

    # Registra los comandos CLI
    app.cli.add_command(init_db_command)

    return app

# Comando CLI para la base de datos
@click.command('init-db')
@with_appcontext
def init_db_command():
    """Limpia los datos existentes y crea tablas nuevas."""
    click.echo('Creando todas las tablas de la base de datos...')
    db.create_all()
    click.echo('Base de datos inicializada.')
