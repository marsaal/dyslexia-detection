# Detecció automàtica de dislèxia a partir de l'àudio de lectura

Sistema d'aprenentatge automàtic per estimar el risc de dislèxia en infants de 3r de primària a partir de gravacions de lectura oral d'un text fix en castellà ("Los Okapis").

El projecte compara tres enfocaments de modelatge —descriptors tabulars, representacions neuronals de l'àudio i una fusió híbrida de totes dues— sobre un conjunt de 141 enregistraments amb fort desequilibri de classes. El detall metodològic i els resultats es documenten a la memòria del TFG.

Els alumnes amb diagnòstic de dislèxia s'identifiquen pel prefix `D_` al nom del fitxer d'àudio.

---

## Estructura del repositori

```
.
├── raw_data/            # Àudios originals (.wav), organitzats per escola i classe
├── processed_data/      # Àudios preprocessats (mateixa estructura que raw_data/)
├── transcriptions/      # Alineacions paraula a paraula generades per error_detection.ipynb
├── datasets/            # Datasets tabulars generats pels notebooks d'extracció
├── outputs/             # Models entrenats, embeddings i resultats d'avaluació
├── utils/               # Mòduls compartits (càrrega de dades, mètriques, validació)
└── requirements.txt     # Dependències Python
```

> **Disponibilitat de les dades.** Els enregistraments contenen dades personals de menors i, per raons de privadesa, no s'inclouen al repositori. Tampoc s'hi distribueixen les dades derivades (`processed_data/`, `transcriptions/`, `datasets/`). L'arbre anterior reflecteix l'estructura local esperada: només es publica el codi.

---

## Notebooks

Els notebooks són seqüencials: cada pas depèn de les sortides de l'anterior i cal executar-los en ordre.

```
error_detection → 01 → 02 → 03 → 04 → 05 → 06 → 07 → 08 → 09
```

| Notebook | Funció | Sortida principal |
|---|---|---|
| `error_detection` | Transcriu els àudios amb Whisper (`large`, `es`) i els alinea paraula a paraula amb el text de referència per programació dinàmica. | `transcriptions/` |
| `01_load_data` | Càrrega dels `.wav` i extracció de metadades del nom i la ruta (escola, classe, etiqueta `D_`, edat, llengua materna); exploració inicial. | `datasets/metadata.csv` |
| `02_preprocessing` | Cadena de preprocessament d'àudio: reducció de soroll, *noise gate*, compressió, equalització, normalització de pic i retall de silencis. | `processed_data/` |
| `03_fluency` | Indicadors de fluïdesa i tempo (temps de lectura i d'articulació, nombre i durada de les pauses). | `datasets/fluency_indicators.csv` |
| `04_errors` | Mètriques d'errors de lectura a partir de l'alineació (substitucions, omissions, addicions, WER, precisió). | `datasets/error_metrics.csv` |
| `05_prosodic` | Anàlisi prosòdica i espectral: dinàmica vocal (Praat/`parselmouth`) i perfils Mel, amb comparació estadística entre grups. | `datasets/prosody_features_extracted.csv` |
| `06_model_tabular` | Entrena i compara quatre models clàssics (LogReg, Random Forest, XGBoost, SVM) sobre les 22 característiques tabulars amb *nested* CV 5×3; selecció per AUC-ROC i interpretabilitat SHAP. | `outputs/holdout_split.json`, `outputs/tabular_model.pkl` |
| `07_model_espectrogrames` | Classificació a partir de l'àudio en brut amb *backbones* preentrenats congelats (PANNs CNN14, WavLM-base, wav2vec2-base) i caps d'agregació MIL; CV a nivell de subjecte. | `outputs/final_model.joblib`, *embeddings* |
| `08_model_hibrid` | Fusiona el model tabular i el neuronal a nivell d'infant i compara estratègies de fusió (*early*, *late* per rang, *late* calibrada i *stacking*). | `outputs/final_hybrid_model.joblib` |
| `09_millor_model` | Identifica el millor model global, l'avalua sobre el *hold-out* (29 infants) i n'analitza els errors (matriu de confusió, corbes ROC/PR i projecció PCA). | `imatges/` |

**Protocol comú.** Tots els blocs comparteixen un protocol de validació únic: CV estratificada 5×3, AUC-ROC com a mètrica de selecció, IC 95 % per *bootstrap* i *permutation test* per a la significació. El llindar de decisió de cada model es deriva maximitzant la F2 sobre les prediccions *out-of-fold*.

**Partició persistent.** La primera execució de `06` genera `outputs/holdout_split.json` amb els índexs exactes de la partició *train*/*hold-out* (80/20 estratificat). Els blocs `07`, `08` i `09` carreguen aquest fitxer, garantint que tots els models s'avaluen sobre exactament les mateixes mostres.

**Model final.** El millor model global és la **fusió *late* calibrada**: combina la regressió logística tabular i el *linear probe* sobre *embeddings* WavLM portant cada branca a probabilitats comparables (sigmoide de Platt amb CV interna) i promitjant-les. Entre estratègies estadísticament indistingibles (IC solapats), s'escull aquesta per ser no transductiva i aplicable cas a cas, requisit d'un cribratge real.

---

## Instal·lació

```bash
python -m venv .venv
.venv\Scripts\activate

pip install -r requirements.txt
```

El notebook `error_detection.ipynb` requereix dependències addicionals, instal·lables des de la seva pròpia cel·la `%pip install`:

```bash
pip install torch openai-whisper whisper-timestamped python-Levenshtein
```

El notebook `07_model_espectrogrames.ipynb` s'ha dissenyat per executar-se a **Google Colab**, ja que els *backbones* de parla requereixen GPU.

---

## Convenció d'etiquetes

| Valor | Significat |
|---|---|
| `dislexia = 1` | Alumne amb diagnòstic de dislèxia (prefix `D_` al nom) |
| `dislexia = 0` | Alumne sense diagnòstic (grup control) |