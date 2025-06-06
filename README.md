# Système de Reconnaissance Vocale Multilingue
## Reconnaissance Automatique de la Parole pour l'Anglais, l'Arabe et le Darija

### 📖 Description du Projet

Ce projet développe des systèmes de reconnaissance automatique de la parole (ASR) adaptés à trois langues distinctes : l'anglais, l'arabe standard moderne (MSA) et le darija (dialecte marocain). Le projet utilise des approches architecturales différenciées selon les ressources linguistiques disponibles.

### 🏗️ Architecture du Projet

#### Approches Utilisées :
- **Pour l'anglais** : Architectures hybrides CNN+LSTM et CNN+GRU inspirées de DeepSpeech2
- **Pour l'arabe et le darija** : Fine-tuning de modèles Wav2Vec 2.0 pré-entraînés

### 📁 Structure des Fichiers

#### 🌐 Interface Utilisateur
- **`app0`** : Application Flask principale pour l'interface web
  - Interface utilisateur permettant de choisir la langue (Anglais, Arabe, Darija)
  - Upload de fichiers audio et affichage des transcriptions
  - Routage automatique vers les modèles spécifiques selon la langue

#### 🤖 Modèles et Chargement
- **`model_loader`** : Script de chargement du modèle anglais
  - Chargement du modèle CNN+GRU entraîné pour l'anglais
  - Préparation et préprocessing des fichiers audio anglais
  - Fonctions de prédiction pour les transcriptions anglaises

#### 🇲🇦 Modèles Darija
- **`model_darija`** : Modèle fine-tuné pour le dialecte darija
  - Modèle Wav2Vec2 spécialisé pour le darija marocain
  - Gestion des spécificités phonétiques du dialecte
- **`wav2vec-darija-processor`** : Processeur pour le darija
  - Préprocessing spécialisé pour les audio en darija
  - Tokenisation et normalisation adaptées au dialecte

#### 🇸🇦 Modèles Arabe Standard
- **`model_arabe`** : Modèle fine-tuné pour l'arabe 
  - Modèle Wav2Vec2 adapté aux spécificités de l'arabe 
  - Gestion de la complexité morphologique de l'arabe
- **`wav2vec2-ar`** : Processeur Wav2Vec2 pour l'arabe
  - Préprocessing et tokenisation pour l'arabe standard

#### 📊 Notebooks de Développement
- **`model_cnn`** : Notebook de comparaison des architectures anglaises
  - Expérimentation et comparaison de CNN+BiLSTM, GRU seul, et CNN+GRU
  - Évaluation des performances (WER/CER)
  - Sélection du meilleur modèle (CNN+GRU avec WER=0.16)

- **`model_arabe`** : Notebook de fine-tuning pour l'arabe standard
  - Fine-tuning du modèle Wav2Vec2 sur le corpus Common Voice arabe
  - Optimisation des hyperparamètres
  - Évaluation des performances

- **`model_darija`** : Notebook de fine-tuning pour le darija
  - Fine-tuning sur le dataset UBC-NLP Casablanca
  - Adaptation aux défis du darija (orthographe non-standardisée, code-switching)

#### 📋 Configuration
- **`templates`** : Templates HTML pour l'interface Flask
  - Interface utilisateur responsive
  - Formulaires d'upload et affichage des résultats

### 🎯 Résultats de Performance

#### Modèles Anglais (sur Common Voice English)
| Modèle | WER | CER |
|--------|-----|-----|
| CNN+GRU | **0.16** | **0.10** |
| CNN+LSTM | 0.29 | 0.15 |
| GRU seul | 0.34 | 0.18 |

#### Modèles Arabes
- **Arabe Standard** : Fine-tuning réussi avec Wav2Vec2-large-xlsr-53-arabic
- **Darija** : Adaptation spécialisée malgré les défis du dialecte

### 🚀 Installation et Utilisation

#### Prérequis
```bash
pip install torch torchaudio transformers
pip install flask tensorflow librosa
pip install datasets evaluate jiwer
```

#### Lancement de l'Application
```bash
python app0.py
```

#### Utilisation des Notebooks
1. Ouvrir `model_cnn.ipynb` pour l'entraînement des modèles anglais
2. Utiliser `model_arabe.ipynb` pour le fine-tuning arabe
3. Exécuter `model_darija.ipynb` pour l'adaptation au darija

### 🔬 Méthodologie

#### Préprocessing Audio
- Rééchantillonnage à 16 kHz
- Extraction de spectrogrammes mel via STFT
- Normalisation et augmentation des données

#### Architectures Utilisées
- **CNN+GRU** : Extraction locale (CNN) + modélisation temporelle (GRU)
- **Wav2Vec2** : Représentations auto-supervisées + fine-tuning CTC

#### Évaluation
- **WER (Word Error Rate)** : Erreur au niveau des mots
- **CER (Character Error Rate)** : Erreur au niveau des caractères
- **CTC Loss** : Fonction de perte pour l'alignement automatique

### 📚 Datasets Utilisés

- **Anglais** : Mozilla Common Voice English (sous-ensemble)
- **Arabe** : Mozilla Common Voice Arabic
- **Darija** : UBC-NLP Casablanca Dataset

### 🔍 Contributions Principales

1. **Approche différenciée** selon les ressources linguistiques
2. **Performance optimale** pour l'anglais avec CNN+GRU
3. **Adaptation réussie** aux variétés arabes avec Wav2Vec2
4. **Interface web multilingue** pour l'utilisation pratique
5. **Évaluation comparative** des architectures

### 🎯 Applications

- Assistants vocaux multilingues
- Systèmes de transcription automatique
- Services d'accessibilité
- Analyse de contenu média arabophone

### 👥 Crédits

**Auteur** : Bouramtane Jihane  
**Superviseur** : Prof. Mohamed Cherradi  
**Institution** : Université Abdelmalek Essaadi, École Nationale des Sciences Appliquées

### 📄 Licence

Ce projet est développé dans un cadre académique de recherche en traitement automatique de la parole.

---

*Pour plus de détails sur la méthodologie et les résultats, consultez le paper de recherche complet.*
