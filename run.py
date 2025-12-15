# ~/SubeTuBanqueta/run.py

from app import create_app
from config import Config
import traceback

# --- INICIO DE MODIFICACIÓN ---
try:
    from import_script import import_batch_command
    cli_command_loaded = True
except ImportError as e:
    cli_command_loaded = False
    print("="*60)
    print("!!! ERROR FATAL AL CARGAR 'import_script.py' !!!")
    print(f"El comando 'import-batch' NO se registrará.")
    print(f"Error específico: {e}")
    print("\nDetalle del error (traceback):")
    traceback.print_exc()
    print("="*60)
except Exception as e:
    cli_command_loaded = False
    print("="*60)
    print("!!! OCURRIÓ UN ERROR INESPERADO AL CARGAR 'import_script.py' !!!")
    print(f"El comando 'import-batch' NO se registrará.")
    print(f"Error: {e}")
    print("\nDetalle del error (traceback):")
    traceback.print_exc()
    print("="*60)
# --- FIN DE MODIFICACIÓN ---


# Creamos la app llamando a nuestra factory
app = create_app(Config)

# --- INICIO DE MODIFICACIÓN ---
# Registrar el comando si se importó correctamente
if cli_command_loaded:
    app.cli.add_command(import_batch_command)
    print("--- Comando 'import-batch' registrado exitosamente ---")
# --- FIN DE MODIFICACIÓN ---


if __name__ == '__main__':
    # Lee el GOOGLE_API_KEY del entorno
    app.run(debug=True, host='0.0.0.0')
