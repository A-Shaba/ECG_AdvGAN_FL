# ECG Adversarial Robustness

> **Adversarial attacks on Deep Learning models for ECG classification**  
> Studio delle vulnerabilità di modelli di deep learning per la classificazione automatica di tracciati ECG, in contesto centralizzato e federated learning, con focus su attacchi *test-time* e *train-time*.

---

## 📋 Descrizione del Progetto

L'obiettivo del progetto è analizzare e valutare la **robustezza** di modelli di classificazione ECG in scenari avversi.  
Il lavoro copre:

- **Attacchi evasion** (*test-time*) → perturbazioni aggiunte agli input per ingannare un modello già addestrato (es. FGSM, PGD, AdvGAN).
- **Attacchi poisoning** (*train-time*) → inserimento di dati falsi plausibili per compromettere l'addestramento (es. VagueGAN).
- **Scenario centralizzato e Federated Learning** → valutazione della sicurezza in contesti distribuiti.
- **Strategie di difesa** → metodi di robustezza a livello di training, preprocessing e detection.

---

## 🏗 Struttura del Repository

```
ecg-adv-robustness/
├── README.md
├── environment.yaml
│
├── data/
│   ├── raw/                    # dati ECG originali (immagini)
│   ├── processed/              # dati preprocessati per il training
│   ├── preprocess_dataset.py   # funzioni di preprocessing e split
│
├── models/
│   ├── ecg_classifier_cnn.py    # modello baseline (CNN/ResNet)
│   ├── utils.py                # utilità per i modelli
│
├── training/
│   ├── train_baseline.py       # training su dati puliti
│   ├── eval_baseline.py        # valutazione modelli
│
├── attacks/
│   ├── gradient_based/         # attacchi basati su gradienti
│   │     ├── fgsm.py
│   │     ├── pgd.py
│   │     ├── cw.py
│   │     ├── deepfool.py
│   │
│   ├── gan_based/
│   │     ├── advGAN/           # attacchi evasion con GAN
│   │     │     ├── generator_2d.py
│   │     │     ├── discriminator_2d.py
│   │     │     ├── advgan.py
│   │     │     ├── train_advgan.py
│   │     │     ├── attack_advgan.py
│   │     │
│   │     ├── poisoningGAN/    # attacchi poisoning con GAN
│   │           ├── generator_2d.py
│   │           ├── discriminator_2d.py
│   │           ├── poisoning_gan.py
│   │           ├── train_poisoning_gan.py
│   │           ├── inject_poisoning.py
│
├── defenses/
│   ├── training_based/
│   │     ├── adversarial_training.py
│   │     ├── def_distillation.py
│   │
│   ├── preprocessing_based/
│   │     ├── filtering.py
│   │     ├── denoising.py
│   │
│   ├── detection_based/
│   │     ├── anomaly.py
│
├── federated/
│   ├── core/
│   │     ├── fl_server.py
│   │     ├── fl_client.py
│   │
│   ├── fl_attack_wrapper.py
│   │
│   ├── fl_defense_wrapper.py
│   │
│   ├── run_federated.py
│
├── experiments/
│   ├── configs/
│   ├── notebooks/
│
└── utils/
    ├── metrics.py
    ├── viz.py
    └── seed.py
```

---

## 📊 Dataset

Il progetto supporta dataset ECG in formato **immagine**.  
Esempi utilizzabili:
- [PTB-XL](https://physionet.org/content/ptb-xl/1.0.1/)
- MIT-BIH arrhythmia dataset (convertito in immagini)

Il preprocessing (`prepreprocess_dataset.py`) include:
- Ridimensionamento e normalizzazione delle immagini
- Conversione a tensori PyTorch
- Creazione di split train/val/test

---

## 🧠 Modelli

### `ecg_classifier_cnn.py`
- Implementazione di una CNN 2D semplice o una ResNet adattata per immagini ECG.
- Configurabile tramite file di configurazione in `experiments/configs/`.

---

## ⚔️ Attacchi Implementati

### Gradient-based (*Evasion*)
- **FGSM** – Fast Gradient Sign Method
- **PGD** – Projected Gradient Descent
- **CW** – Carlini & Wagner
- **DeepFool**

### GAN-based
- **AdvGAN** (*Evasion*) – Generatore di perturbazioni impercettibili per immagini ECG.
- **VagueGAN** (*Poisoning*) – Generatore di dati plausibili ma falsi per corrompere il dataset di training.

---

## 🛡 Difese Implementate

### Training-based
- **Adversarial Training** – retraining del modello con campioni avversari
- **Defensive Distillation** – riduce la sensibilità ai gradienti

### Preprocessing-based
- **Filtering** – filtri passa-basso, smoothing
- **Denoising** – autoencoder di denoising, trasformate wavelet

### Detection-based
- **Anomaly Score** – calcolo di score statistici per rilevare input sospetti

---

## 🤝 Federated Learning

La cartella `federated/` contiene:
- **Server FL** – aggregazione dei modelli
- **Client FL** – addestramento locale
- **Wrapper di attacco/difesa** – adattamento di attacchi e difese al contesto federato
- **Simulazioni** – avvio di esperimenti FL con client benigni e malevoli

---

## 🚀 Come iniziare

### 1. Creare l'ambiente
```bash
conda env create -f environment.yaml
conda activate ecg-adv-robustness
```

### 2. Preparare i dati
```bash
python data/preprocess_dataset.py --input data/raw --output data/processed
```

### 3. Addestrare modello baseline
```bash
python training/train_baseline.py --config experiments/configs/baseline.yaml
```

### 4. Lanciare un attacco FGSM
```bash
python attacks/gradient_based/fgsm.py --model models/ecg_classifier_cnn.py --data data/processed
```

### 5. Addestrare AdvGAN
```bash
python attacks/gan_based/advgan/train_advgan.py --config experiments/configs/advgan.yaml
```

### 6. Simulare FL con attacco Poisoning
```bash
python federated/run_federated.py --config experiments/configs/fl_poisoning.yaml
```

---

## 📈 Metriche

Le metriche principali calcolate in `utils/metrics.py`:
- Accuratezza
- Precisione / Recall / F1-score
- Robust Accuracy (su campioni avversari)
- Tasso di rilevamento anomalie (per difese detection-based)

---

## 📌 Note
- Questo progetto è a scopo **accademico**.  
- Gli attacchi avversari qui implementati devono essere usati solo in contesti controllati e per ricerca.

---
