# Detecció automàtica de dislèxia a partir de l'àudio de lectura

Sistema d'aprenentatge automàtic per detectar indicadors de dislèxia en nens de 3r de primària a partir de gravacions de lectura oral en castellà ("Los okapis").

Els alumnes amb diagnòstic de dislèxia s'identifiquen per el prefix `D_` al nom del fitxer d'àudio.

---

## Estructura del repositori

```
.
├── raw_data/               # Àudios originals (.wav), organitzats per escola i classe
├── processed_data/         # Àudios preprocessats (mateixa estructura que raw_data/)
├── transcriptions/         # CSVs d'alineació paraula a paraula generats per error_detection.ipynb
├── datasets/               # Datasets tabulars generats pels notebooks d'extracció
├── outputs/                # Models entrenats, embeddings i resultats d'avaluació
├── utils/                  # Mòduls Python compartits pels notebooks de modelatge
│   ├── data_loading.py     # Càrrega del dataset tabular i contracte de divisió train/hold-out
│   ├── metrics.py          # Càlcul de mètriques (F1, AUC-ROC, AUC-PR, especificitat…)
│   └── validation.py       # Nested cross-validation i helpers d'avaluació
├── imatges/                # Figures exportades en PDF/PNG
├── streamlit_app/          # Aplicació de demostració (Streamlit) que carrega els models d'outputs/
└── requirements.txt        # Dependències Python
```

---

## Notebooks

Els notebooks segueixen un ordre seqüencial. Cal executar-los en el mateix ordre per garantir que les dependències entre fitxers es compleixen.

```
error_detection.ipynb  →  01  →  02  →  03  →  04  →  05  →  06  →  07  →  08  →  09
```

### Pas previ — `error_detection.ipynb`

**Generació de les transcripcions ASR**

Transcriu els àudios de `raw_data/` amb el model **Whisper** (`large`, en mode `es`) i alinea cada paraula reconeguda amb el text de referència ("Los okapis") mitjançant programació dinàmica (distància d'edició). Per a cada paraula del text de referència produeix una operació d'edició (`correct`, `substitution`, `deletion` o `insertion`) amb el temps de pronúncia associat.

- **Entrada**: àudios de `raw_data/` + text de referència (`text.txt`)
- **Sortida**: fitxers CSV a `transcriptions/` (un per alumne), usats per `04_errors.ipynb`

---

### `01_load_data.ipynb` — Càrrega i exploració de dades

Recorre recursivament tots els `.wav` de `raw_data/` i extreu les metadades codificades al nom del fitxer i la ruta:

| Camp | Descripció |
|---|---|
| `escola`, `classe` | Extrets de la ruta del directori |
| `dislexia` | Prefix `D_` al nom → etiqueta binària |
| `nom`, `edat` | Nom i edat de l'alumne |
| `text_llegit` | Text de lectura (sempre "Los okapis") |
| `llengua_materna` | Llegit del nom del fitxer |

Inclou anàlisi exploratòria inicial: distribució de grups, durades dels àudios i visualitzacions.

- **Sortida**: `datasets/metadata.csv`

---

### `02_preprocessing.ipynb` — Preprocessament dels àudios

Aplica una cadena de processament d'àudio a cada fitxer de `raw_data/` i desa el resultat a `processed_data/` mantenint l'estructura de directoris:

1. Reducció de soroll estacionari (`noisereduce`, 80%)
2. Noise Gate (−30 dB) per eliminar respiracions i sorolls de fons
3. Compressor dinàmic (threshold −16 dB, ràtio 4:1)
4. Equalització de la veu (+10 dB a 400 Hz, +5 dB a 4 kHz)
5. Normalització d'amplitud
6. Retall de silencis d'inici i final

- **Entrada**: `raw_data/`
- **Sortida**: `processed_data/`

---

### `03_fluency.ipynb` — Indicadors de fluïdesa

Extreu indicadors de fluïdesa temporal a partir dels segments de parla detectats amb `librosa.effects.split` (llindar −30 dB):

| Indicador | Descripció |
|---|---|
| `t_total` | Durada total de la lectura (s) |
| `t_art` | Temps efectiu de parla, sense silencis (s) |
| `r_art` | Ràtio de fonació: `t_art / t_total` |
| `n_pauses` | Nombre de pauses detectades |
| `m_pauses` | Durada mitjana de les pauses (s) |
| `t_pauses` | Durada total de les pauses (s) |

Inclou comparació estadística entre grups (t-test de Welch) i visualitzacions.

- **Entrada**: `processed_data/`
- **Sortida**: `datasets/fluency_indicators.csv`

---

### `04_errors.ipynb` — Mètriques d'errors de lectura

Agrega les operacions d'edició dels CSVs de `transcriptions/` i calcula per alumne:

| Mètrica | Descripció |
|---|---|
| `n_correct` | Nombre de paraules llegides correctament |
| `n_sub` | Nombre de substitucions |
| `n_omis` | Nombre d'omissions (`deletion`) |
| `n_add` | Nombre d'addicions (`insertion`) |
| `n_err` | Nombre total d'errors |
| `wer_lect` | Word Error Rate (% d'errors sobre paraules objectiu) |
| `acc_lect` | % de paraules llegides correctament |
| `n_conf` | Paraules amb temps per caràcter anormalment alt (z-score > 2) |

Inclou comparació entre grups i visualitzacions de la distribució per tipus d'error.

- **Entrada**: `transcriptions/`
- **Sortida**: `datasets/error_metrics.csv`

---

### `05_prosodic.ipynb` — Anàlisi espectral i prosòdica

Dos blocs d'anàlisi sobre els àudios processats:

**Bloc 1 — Perfil Mel (128 bandes, FFT 1024, hop 256)**
Compara l'energia espectral banda a banda entre grups (Dislèxia vs. Control) mitjançant:
- Perfil Mel mitjà per grup i diferències per banda
- Tests de Welch per banda amb correcció de Bonferroni i FDR-BH
- Mida d'efecte (Cohen's d)
- Volcano plot
- Comparació per macrobandes (baixa / mitjana / alta)

**Bloc 2 — Característiques prosòdiques clíniques (Praat / parselmouth)**
Extreu i compara entre grups:
- `f0_mean`, `f0_std` — pitch (freqüència fonamental)
- `int_mean`, `int_std` — intensitat de la veu (dB)

Inclou test de Welch + FDR-BH i mida d'efecte de Hedges g.

- **Entrada**: `processed_data/`
- **Sortida**: `datasets/prosody_features_extracted.csv`

---

### `06_model_tabular.ipynb` — Models clàssics de ML

Entrena i compara quatre models clàssics d'aprenentatge automàtic sobre les característiques tabulars (fluïdesa + errors + prosòdiques):

| Model | Justificació |
|---|---|
| Regressió Logística (L1/L2) | Baseline interpretable |
| Random Forest | Robust al desequilibri de classes |
| XGBoost | Estat de l'art en dades tabulars |
| SVM (kernel RBF) | Bo en conjunts petits d'alta dimensió |

**Protocol de validació**: Nested CV 5×3 amb 3 llavors aleatòries.  
**Mètriques**: F1, F2, Recall, Especificitat, AUC-ROC, AUC-PR, Balanced Accuracy.  
**Selecció del model**: AUC-ROC de CV. **Llindar de decisió**: màxim F2 (β=2) sobre les prediccions out-of-fold.  
**Interpretabilitat**: anàlisi SHAP del model guanyador.

> **Important**: la primera execució d'aquest notebook genera `outputs/holdout_split.json` amb els índexs exactes de la partició train/hold-out (80/20). Els notebooks `07` i `08` carreguen aquest fitxer per garantir que tots els models s'avaluen sobre exactament les mateixes mostres.

- **Entrada**: `datasets/`
- **Sortida**: `outputs/holdout_split.json`, `outputs/tabular_model.pkl`, `outputs/tabular_harmon_comparison.csv`

---

### `07_model_espectrogrames.ipynb` — Models basats en espectrogrames (deep learning)

Dues etapes unificades per a la classificació a partir de l'àudio en brut:

**Part 1 — Generació de les dades**
Fragmenta cada àudio (~66 s) en finestres de **5 s amb solapament del 50% (stride 2,5 s)** (~25 fragments per alumne). Genera i desa en memòria cau:
- `outputs/mel_fragments.npz` — espectrogrames Mel per a tots els fragments
- `outputs/emb_PANNs.npy`, `outputs/emb_wav2vec2.npy`, `outputs/emb_WavLM.npy` — embeddings pre-computats amb cada backbone

**Part 2 — Selecció del millor model**
Compara *backbones* pre-entrenats congelats:
- **PANNs CNN14** (àudio general, AudioSet)
- **WavLM-base** (parla, Microsoft)
- **wav2vec2-base** (parla, Meta)

I caps de classificació (*heads*):
- Linear probe
- Mean-pool MIL
- Gated-Attention MIL

**Protocol de validació**: CV estratificada a nivell de subjecte (cap fragment d'un alumne apareix alhora a train i validació) + IC 95% per bootstrap + permutation test.

- **Entrada**: `processed_data/`, `outputs/holdout_split.json`
- **Sortida**: `outputs/backbone_comparison.csv`, `outputs/head_comparison.csv`, `outputs/emb_*.npy`, `outputs/mel_fragments.npz`, `outputs/final_model.joblib`, `outputs/final_model_holdout.json`

---

### `08_model_hibrid.ipynb` — Model híbrid (fusió tabular + àudio)

Combina els dos models base seleccionats als blocs anteriors —la Regressió Logística tabular (`outputs/tabular_model.pkl`) i el *linear probe* sobre embeddings WavLM (`outputs/final_model.joblib`)— en un únic classificador a nivell d'infant. Les dues modalitats s'alineen per la clau composta `nom|escola|classe`.

Compara cinc estratègies amb el mateix protocol (CV de subjecte 5×3 + IC 95% bootstrap + permutation test):

| Estratègia | On combina | Com |
|---|---|---|
| Només tabular | — | baseline del bloc 06 |
| Només àudio | — | baseline del bloc 07 |
| Fusió *early* | característiques | `concat(tabular, àudio)` → LogReg L2 |
| Fusió *late* (mitjana) | decisió | mitjana dels **rangs percentils** de les dues probabilitats |
| Fusió *late* (*stacking*) | decisió | meta-LogReg sobre les probabilitats base (*cross-fit*) |

Com que els dos models base viuen en espais de probabilitat no comparables, la fusió *late* es fa a nivell de **rang**, no de probabilitat crua. **Selecció**: AUC-ROC de CV; **llindar**: màxim F2 sobre OOF.

- **Entrada**: `outputs/tabular_model.pkl`, `outputs/final_model.joblib`, `outputs/emb_WavLM.npy`, `outputs/mel_fragments.npz`, `outputs/holdout_split.json`
- **Sortida**: `outputs/hybrid_comparison.csv`, `outputs/final_hybrid_model.joblib`, `outputs/final_hybrid_holdout.json`

---

### `09_millor_model.ipynb` — Millor model global i anàlisi d'errors

Tanca el cicle de modelització identificant el **millor model a nivell global** (entre els tres blocs) i analitzant en detall els errors sobre el conjunt de hold-out:

1. **Comparació global** — taula consolidada dels tres blocs amb AUC-ROC de CV, IC 95%, p-valor i AUC-ROC hold-out.
2. **Prediccions del millor model** sobre els 29 infants del hold-out (model híbrid, fusió *late* per rangs, llindar derivat del màxim F2 sobre OOF).
3. **Anàlisi d'errors per subjecte** — taula amb nom, escola, classe, probabilitat predita i tipus d'error (FP / FN).
4. **Visualitzacions** — matriu de confusió, corba ROC, corba PR i projecció PCA (espai tabular) marcant els errors.

> Les seccions d'anàlisi d'errors i PCA que originalment eren al bloc 06 s'han mogut aquí per tenir una visió unificada del millor model.

- **Entrada**: `outputs/final_hybrid_model.joblib`, `outputs/final_hybrid_holdout.json`, `outputs/final_model_holdout.json`, `outputs/tabular_harmon_comparison.csv`, `outputs/emb_WavLM.npy`, `outputs/mel_fragments.npz`, `outputs/holdout_split.json`
- **Sortida**: `imatges/millor_model_holdout_eval.pdf`, `imatges/millor_model_pca_holdout.pdf`

---

## Flux de dades

```
raw_data/
    │
    ├──► error_detection.ipynb ──────────────────► transcriptions/
    │
    ├──► 01_load_data.ipynb ─────────────────────► datasets/metadata.csv
    │
    └──► 02_preprocessing.ipynb ─────────────────► processed_data/
                │
                ├──► 03_fluency.ipynb ───────────► datasets/fluency_indicators.csv
                │
                ├──► 04_errors.ipynb ────────────► datasets/error_metrics.csv
                │        ▲
                │        └── transcriptions/
                │
                └──► 05_prosodic.ipynb ──────────► datasets/prosody_features_extracted.csv

datasets/ ──► 06_model_tabular.ipynb ────────────► outputs/ (tabular_model.pkl + holdout_split.json)

processed_data/ + outputs/holdout_split.json
    └──► 07_model_espectrogrames.ipynb ──────────► outputs/ (embeddings + final_model.joblib)

outputs/tabular_model.pkl + final_model.joblib + emb_WavLM.npy
    └──► 08_model_hibrid.ipynb ──────────────────► outputs/ (final_hybrid_model.joblib)

outputs/final_hybrid_model.joblib + *_holdout.json + emb_WavLM.npy
    └──► 09_millor_model.ipynb ──────────────────► imatges/ (avaluació global + errors)
```

---

## Instal·lació

```bash
python -m venv .venv
.venv\Scripts\activate        # Windows
# source .venv/bin/activate   # Linux / macOS

pip install -r requirements.txt
```

Per al notebook `error_detection.ipynb` es requereix instal·lació addicional (s'executa la cel·la `%pip install` del propi notebook):

```bash
pip install torch openai-whisper whisper-timestamped python-Levenshtein
```

El notebook `07_model_espectrogrames.ipynb` s'ha dissenyat per executar-se a **Google Colab** (requereix GPU per als backbones de parla).

---

## Aplicació de demostració (Streamlit)

`streamlit_app/app.py` és una aplicació interactiva que carrega els models entrenats desats a `outputs/` (tabular, espectral i híbrid) per explorar-ne les prediccions i analitzar nous àudios de lectura.

```bash
pip install -r streamlit_app/requirements.txt
streamlit run streamlit_app/app.py
```

---

## Convenció d'etiquetes

| Valor | Significat |
|---|---|
| `dislexia = 1` | Alumne amb diagnòstic de dislèxia (prefix `D_` al nom) |
| `dislexia = 0` | Alumne sense diagnòstic (grup control) |
