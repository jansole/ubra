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

"""
Justificació de les dades:
─────────────────────────────────────────────────────────────────────
JUSTIFICACIÓ DELS RANGS (per incloure al TFG)
─────────────────────────────────────────────────────────────────────
 
TEMPERATURA CUTÀNIA MAMÀRIA
  Font: Lubkowska & Chudecka (2021), Int J Environ Res Public Health,
        PMC7908199. DOI: 10.3390/ijerph18031097
  Rang normal (sensor contacte): 33.5 - 36.5 °C (mitjana ~34.6 °C)
  Asimetria fisiològica L/R: 0 - 0.5 °C (patològic > 0.5 °C)
 
  Variació per cicle menstrual (BBT):
  Font: Reed & Carr (2015), StatPearls, NCBI NBK546686
        Clue Health (Freundl et al., 2014), Eur J Contracept
  Fase fol·licular : 36.1 - 36.4 °C (BBT basal, dominància estrògens)
  Post-ovulació    : pujada de 0.3 - 0.6 °C per acció de la progesterona
  Fase lútia       : 36.4 - 37.0 °C (sostre fins menstruació)
  NOTA: El sensor d'Ubra mesura temperatura cutània local (no BBT axil·lar),
  per tant el rang absolut segueix Lubkowska (~33.5-36.5 °C) però el
  PATRÓ bifàsic és el de BBT.
 
BIOIMPEDÀNCIA (BioZ)
  Font: Zou & Guo (2003), Breast Cancer Res Treat, PMC400648
        Arabsalmani et al. (2025) Bioengineering 12(5):521, PMC12109311
        Perlet et al. (citat a Zou & Guo): variació cíclica hormonal
        Jossinet (1996): teixit sa vs. maligne, Clin Phys Physiol Meas
  Rang teixit sa (baixa freqüència, ~1 kHz): 40 - 120 Ω
  Quadrant UO té lleugerament més teixit glandular → impedàncies menors
  Fase lútia: retenció líquids → lleugera baixada d'impedància (-3 a -8 Ω)
  Maligne: impedància < teixit normal (major conductivitat)
 
HUMITAT / SUOR (interfície pell-sensor)
  Font: Richetti et al. (2017), Sensors Actuat, ScienceDirect
        (taxa de suor 500-700 mL/dia en repòs a 25 °C, 50% HR ambient)
  Repòs    : ~30-50 % HR local
  Activitat: ~50-75 % HR local
  Son      : ~40-60 % HR local (lleugerament elevat per oclusió del wearable)
 
FREQÜÈNCIA CARDÍACA (BPM / PPG)
  Font: Tanaka et al. (2001), JACC: FC màxima = 208 - 0.7 x edat
        Schweizer et al. (2025) JMIR Cardio, DOI: 10.2196/67110 
        García et al. (2024) Sensors 24(21):6826
        FC repòs normal adults: 60-100 bpm (AHA guidelines)
  Son      : 50-65 bpm (predominança parasimpàtica nocturna)
  Repòs    : 60-80 bpm
  Activitat: 90-140 bpm (exercici moderat, ~60-70% FC màxima)
 
IMU (RMS acceleració)
  Font: Clasificació activitat física wearables (Troiano et al., 2014)
  Son/repòs: < 0.05 g (quasi estàtic)
  Caminant : 0.2-0.5 g
  Exercici  : 0.5-2.0 g
─────────────────────────────────────────────────────────────────────
"""


RANDOM_SEED = 10102004
np.random.seed(RANDOM_SEED)
 
START_DATE = datetime(2026, 1, 1, 0, 0, 0) # comencem l'1 de gener d'aquest any
DAYS = 30
PATIENT_ID = "PAT_001"
 
# Sensors: 2 mames × 4 quadrants
MAMES = ["E", "D"]
QUADRANTS = ["UO", "UI", "LO", "LI"]
SENSOR_IDS = [f"{m}_{q}" for m in MAMES for q in QUADRANTS]  # 8 sensors


def cicle():
    # Cicle menstrual pot variar entre 3 i 7 dies, s'ha de tenir en compte
    # La fase luteal es considera constant en 14 dies
    menstrual = np.random.randint(3, 8)
    luteal = 14
    cycle_length = 30 # per a simular 1 mes
    ovulation_day = cycle_length - luteal

    labels = []
    for day in range(1, cycle_length + 1):
        if day <= menstrual:
            labels.append("menstrual")
        elif day == ovulation_day:
            labels.append("ovulation")
        elif day > ovulation_day:
            labels.append("luteal")
        else:
            labels.append("follicular")

    return {
        "menstrual_days": menstrual,
        "luteal_days": luteal,
        "cycle_length": cycle_length,
        "ovulation_day": ovulation_day,
        "phase_labels": labels,
    }


# ── Temperatura
# Base cutània mamària: 34.6 °C (Lubkowska & Chudecka, 2021)
# Offset BBT per fase (Reed & Carr, 2015)
TEMP_PHASE_OFFSET = {
    "menstrual"  : (-0.15,  0.05),   # (min_offset, max_offset)
    "follicular" : (-0.15,  0.05),
    "ovulation"  : ( 0.10,  0.25),   # pujada característica pre/post-ovulació
    "luteal"     : ( 0.25,  0.50),   # sosté per progesterona
}

# Mitjanes observades (sensor local) per mama i quadrant
# Dades d'exemple amb valors de l'estudi o descripció proporcionats.
# Right breast = mama dreta (D), Left breast = mama esquerra (E)
MEAN_TEMPS_BY_QUADRANT = {
    "D": {"UO": 32.60, "UI": 32.91, "LO": 32.28, "LI": 33.29},
    "E": {"UO": 32.46, "UI": 32.69, "LO": 32.12, "LI": 33.00},
}

BREAST_ASYMMETRY_THRESHOLD = 0.5
QUADRANT_ASYMMETRY_THRESHOLD = 1.0

def breast_mean_temperatures():
    return {
        m: np.mean([MEAN_TEMPS_BY_QUADRANT[m][q] for q in QUADRANTS])
        for m in MAMES
    }

def breast_quadrant_temperature_differences():
    return {
        q: abs(MEAN_TEMPS_BY_QUADRANT["D"][q] - MEAN_TEMPS_BY_QUADRANT["E"][q])
        for q in QUADRANTS
    }

def whole_breast_asymmetry():
    breast_means = breast_mean_temperatures()
    return abs(breast_means["D"] - breast_means["E"])

def is_abnormal_asymmetry():
    breast_diff = whole_breast_asymmetry()
    quadrant_diffs = breast_quadrant_temperature_differences()
    return {
        "whole_breast": breast_diff > BREAST_ASYMMETRY_THRESHOLD,
        "whole_breast_diff": breast_diff,
        "quadrants": {
            q: diff > QUADRANT_ASYMMETRY_THRESHOLD
            for q, diff in quadrant_diffs.items()
        },
        "quadrant_diffs": quadrant_diffs,
    }

