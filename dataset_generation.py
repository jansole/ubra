import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import os

#Comencem amb la generació de les dades d'una sola pacient

"""
Dades a considerar:
- Giroscopi :        X, Y, Z             (cada 15 segons, agregat)
- Bioimpedancia :    valor global?       (un cop per setmana)
- Temperatura :      corporal en ºC      (cada 1 minut)
- Humitat :          en %                (cada 5 minuts)
- PPG :              registre del BPM    (finestres de 30 segons)
- Autoregistrat :    qüestionari         (cada 2 setmanes a la usuària (o cada una?))

Per cadascuna de les primeres 4:
- Mama :             L/R
- Quadrant :         UO (upper-outer), UI (upper-inner), LO (lower-outer), LI (lower-inner)
"""

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
 
START_DATE = datetime(2026, 1, 1, 0, 0, 0)
DAYS = 30
PATIENT_ID = "PAT_001"
 
# Sensors: 2 mames × 4 quadrants
MAMES = ["E", "D"]
QUADRANTS = ["UO", "UI", "LO", "LI"]
SENSOR_IDS = [f"{m}_{q}" for m in MAMES for q in QUADRANTS]  # 8 sensors


def cicle():
    # Cicle menstrual pot variar entre 3 i 7 dies, s'ha de tenir en compte
    # La luteal sol ser constant en 14 dies
    menstrual = np.random.randint(3,8)

