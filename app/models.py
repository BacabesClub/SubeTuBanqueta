# app/models.py
from app.extensions import db
from datetime import datetime

# RENOMBRADA: de Image a Report
class Report(db.Model):
    # RENOMBRADA: de 'images' a 'reports'
    __tablename__ = 'reports'
    
    id = db.Column(db.Integer, primary_key=True)
    file_name = db.Column(db.String(300), nullable=False)
    location = db.Column(db.String(100), nullable=False)
    date = db.Column(db.DateTime, nullable=False, default=datetime.utcnow)
    observations = db.Column(db.Text, nullable=True)

    # --- NUEVA COLUMNA ---
    # Para saber qué tipo de reporte es: 'accessibility' u 'obstacle'
    report_type = db.Column(db.String(50), nullable=False, default='accessibility')

    # Columnas específicas de accesibilidad (serán NULL para obstáculos)
    user_accessibility = db.Column(db.String(50), nullable=True)
    model_accessibility = db.Column(db.String(50), nullable=True)

    def __repr__(self):
        # Actualizado
        return f'<Report {self.id}: {self.file_name} ({self.report_type})>'

    def to_dict(self):
        # 1. Valores por defecto (Océano Atlántico, cerca de África)
        lat = 0.0
        lng = 0.0

        # 2. Intentamos limpiar el dato sucio
        # Solo entramos si location tiene datos y una coma
        if self.location and ',' in self.location:
            try:
                # Quitamos espacios en blanco
                clean_loc = self.location.replace("unknown", "").strip()

                parts = clean_loc.split(',')
                # Solo convertimos si tenemos las dos partes
                if len(parts) >= 2 and parts[0] and parts[1]:
                    lat = float(parts[0])
                    lng = float(parts[1])
            except ValueError:
                # Si falla la conversión (ej: texto raro), se queda en 0.0
                pass

        return {
            "id": self.id,
            "file_name": self.file_name,
            "location": self.location, # El original (sucio)

            # --- NUEVOS CAMPOS LIMPIOS PARA EL MAPA ---
            "lat": lat,
            "lng": lng,
            # ------------------------------------------

            "date": self.date.isoformat(),
            "observations": self.observations or "",
            "report_type": self.report_type,
            "accessibility": self.user_accessibility or "No especificado",
            "model_accessibility": self.model_accessibility or ""
        }
