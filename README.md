# ButterflyAnevrisme - Graph Neural Networks pour la Simulation d'Anévrismes

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.9+-red.svg)](https://pytorch.org/)
[![PyTorch Geometric](https://img.shields.io/badge/PyTorch%20Geometric-2.0+-green.svg)](https://pytorch-geometric.readthedocs.io/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

## 🎯 Overview

Ce projet implémente un système de simulation d'anévrismes cérébraux utilisant des Graph Neural Networks (GNNs) avec une architecture Encode-Process-Decode. Le projet démontre l'application de techniques avancées de deep learning pour la modélisation physique et médicale, spécifiquement pour la prédiction du comportement mécanique des vaisseaux sanguins.

## 🚀 Key Features

### Architecture Graph Neural Network
- **Encode-Process-Decode** - Architecture spécialisée pour les données graphiques
- **Message Passing** - 15 couches de passage de messages pour la propagation d'information
- **GraphNet Blocks** - Blocs de traitement optimisés pour les réseaux de graphes
- **Multi-Layer Perceptrons** - MLPs avec Layer Normalization pour l'encodage/décodage

### Simulation Physique
- **Modélisation d'Anévrismes** - Simulation du comportement mécanique des vaisseaux
- **Données Géométriques** - Traitement de maillages 3D et données structurales
- **Prédiction de Déformation** - Prédiction des déformations sous charge
- **Analyse de Contraintes** - Évaluation des contraintes mécaniques

### Traitement des Données
- **DataLoader Personnalisé** - Chargement optimisé des données de maillage
- **Préprocessing** - Utilitaires de traitement et normalisation
- **Visualisation** - Outils d'analyse exploratoire des données (EDA)

## 📁 Project Structure

```
ButterflyAnevrisme/
├── Train.py                    # Script principal d'entraînement
├── EncoderDecoder.py           # Architecture Encode-Process-Decode
├── MessagePassing.py           # Implémentation du passage de messages
├── DataLoader.py               # Chargement et préprocessing des données
├── Utils.py                    # Utilitaires et fonctions auxiliaires
├── Epoch.py                    # Gestion des époques d'entraînement
├── processing_utils.py         # Utilitaires de traitement
├── data_analysis/              # Analyse exploratoire des données
│   ├── eda1_tanguy.ipynb       # Notebook d'analyse exploratoire
│   ├── eda1_tanguy_results.pdf # Résultats de l'analyse
│   └── Paroi_results.pdf       # Analyse des parois vasculaires
└── README.md                   # Documentation du projet
```

## 🛠️ Installation

### Prérequis
- Python 3.8+
- PyTorch 1.9+
- PyTorch Geometric 2.0+
- CUDA (optionnel, pour l'accélération GPU)
- Meshio (pour le traitement des maillages)

### Installation des dépendances

```bash
# Cloner le repository
git clone https://github.com/arthurriche/ButterflyAnevrisme.git
cd ButterflyAnevrisme

# Installer PyTorch (ajuster selon votre version CUDA)
pip install torch torchvision torchaudio

# Installer PyTorch Geometric
pip install torch-geometric

# Installer les dépendances supplémentaires
pip install meshio
pip install tensorboard
pip install numpy
pip install matplotlib
pip install scipy
```

## 📈 Quick Start

### 1. Préparation des Données

```python
# Charger le dataset
from DataLoader import Dataset
folder_path = '/path/to/your/data/'
dataset = Dataset(folder_path)
```

### 2. Configuration du Modèle

```python
from EncoderDecoder import EncodeProcessDecode

model = EncodeProcessDecode(
    node_input_size=6,      # Taille des features des nœuds
    edge_input_size=3,      # Taille des features des arêtes
    message_passing_num=15, # Nombre de couches de message passing
    hidden_size=128,        # Taille des couches cachées
    output_size=3,          # Taille de la sortie
)
```

### 3. Entraînement

```python
# Lancer l'entraînement
python Train.py
```

## 🧮 Technical Implementation

### Architecture Encode-Process-Decode

```python
class EncodeProcessDecode(nn.Module):
    def __init__(self, message_passing_num, node_input_size, 
                 edge_input_size, output_size, hidden_size=128):
        super().__init__()
        
        # Encoder: Transforme les features d'entrée en représentation latente
        self.encoder = Encoder(node_input_size, edge_input_size, hidden_size)
        
        # Processor: Passage de messages entre les nœuds
        self.processer_list = nn.ModuleList([
            GraphNetBlock(hidden_size) for _ in range(message_passing_num)
        ])
        
        # Decoder: Génère la sortie finale
        self.decoder = Decoder(hidden_size, output_size)
```

### Message Passing

Le système utilise 15 couches de message passing pour propager l'information à travers le graphe :

```python
# Propagation de l'information
for processor in self.processer_list:
    graph = processor(graph)
```

### Traitement des Maillages

- **Conversion de Formats** - Support pour différents formats de maillage
- **Extraction de Features** - Calcul des caractéristiques géométriques
- **Normalisation** - Standardisation des données d'entrée

## 📊 Performance Metrics

### Métriques d'Entraînement
- **Loss Function** - L2 Loss pour la régression
- **Training Loss** - Suivi de la perte d'entraînement
- **Validation Metrics** - Évaluation sur l'ensemble de validation
- **TensorBoard** - Visualisation en temps réel des métriques

### Métriques de Simulation
- **Précision Géométrique** - Exactitude de la prédiction de déformation
- **Convergence Physique** - Stabilité de la simulation
- **Temps de Calcul** - Performance computationnelle

## 🔬 Advanced Features

### Optimisation GPU
- **CUDA Support** - Accélération GPU pour l'entraînement
- **Memory Management** - Gestion optimisée de la mémoire
- **Batch Processing** - Traitement par lots pour l'efficacité

### Analyse des Données
- **Exploratory Data Analysis** - Notebooks d'analyse exploratoire
- **Visualisation 3D** - Rendu des maillages et résultats
- **Statistical Analysis** - Analyse statistique des résultats

### Modélisation Physique
- **Contraintes Mécaniques** - Respect des lois de la mécanique
- **Conditions aux Limites** - Gestion des conditions aux frontières
- **Validation Physique** - Vérification de la cohérence physique

## 🚀 Applications

### Médical
- **Diagnostic Assisté** - Aide au diagnostic d'anévrismes
- **Planification Chirurgicale** - Simulation préopératoire
- **Évaluation des Risques** - Prédiction du risque de rupture

### Recherche
- **Modélisation Biomécanique** - Étude du comportement vasculaire
- **Développement de Prothèses** - Optimisation des implants
- **Validation de Protocoles** - Test de nouveaux protocoles

## 📚 Documentation Technique

### Architecture du Modèle
- **Encoder** - Transformation des features d'entrée
- **Processor** - Propagation d'information dans le graphe
- **Decoder** - Génération des prédictions finales

### Traitement des Données
- **Format des Données** - Structure des fichiers d'entrée
- **Préprocessing** - Étapes de préparation des données
- **Augmentation** - Techniques d'augmentation de données

### Hyperparamètres
- **Learning Rate** - 0.0001 (Adam optimizer)
- **Batch Size** - 1 (traitement séquentiel)
- **Hidden Size** - 128 (taille des couches cachées)
- **Message Passing Layers** - 15 (profondeur du réseau)

## 🤝 Contributing

1. Fork le repository
2. Créer une branche feature (`git checkout -b feature/amazing-feature`)
3. Commit les changements (`git commit -m 'Add amazing feature'`)
4. Push vers la branche (`git push origin feature/amazing-feature`)
5. Ouvrir une Pull Request

## 📝 License

Ce projet est sous licence MIT - voir le fichier [LICENSE](LICENSE) pour plus de détails.

## 👨‍💻 Author

**Arthur Riche**
- LinkedIn: [Arthur Riche]([https://www.linkedin.com/in/arthurriche/](https://www.linkedin.com/in/arthur-riché-7a277719a/))
- Email: arthur57140@gmail.com

## 🙏 Acknowledgments

- **Équipe de Recherche** pour les données et la supervision
- **PyTorch Geometric** pour les outils de GNN
- **Communauté Open Source** pour les bibliothèques utilisées
- **Institutions Médicales** pour la validation clinique

---

⭐ **Star ce repository si vous le trouvez utile !**

*Ce projet démontre l'application avancée des Graph Neural Networks pour la simulation biomécanique et la modélisation médicale.* 
