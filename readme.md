📑 Documentation du Notebook : Entraînement YOLOv8 (SKU Vision Pro)
Ce répertoire contient le notebook principal utilisé pour l'entraînement, la validation et l'évaluation du modèle YOLOv8 dans le cadre du projet Automatic Retail Shelf Monitoring.

🎯 Objectif du Notebook
L'objectif est de transformer le dataset brut SKU-110K en un moteur d'inférence capable de détecter des produits dans des environnements de vente à haute densité avec une précision supérieure à 85 %.

🛠️ Prérequis Techniques
Le notebook est optimisé pour s'exécuter sur Kaggle ou Google Colab avec l'environnement suivant :

GPU : NVIDIA Tesla T4 (minimum 16 Go VRAM recommandé).

Framework : PyTorch 2.x.

Librairie IA : Ultralytics YOLOv8.

Traitement d'images : OpenCV et Albumentations.

📥 Préparation du Dataset (SKU-110K)
Le notebook automatise les étapes suivantes :

Extraction : Décompression des images et des fichiers de labels.

Formatage : Conversion des annotations du format CSV/Global au format .txt spécifique à YOLO (normalized x_center, y_center, width, height).

Structuration : Organisation des dossiers en train/, val/, et test/.

⚙️ Configuration de l'Entraînement
Le modèle utilisé est YOLOv8n (Nano) pour garantir une vitesse d'inférence compatible avec le temps réel.

Paramètres clés :
Époques : 50 (avec une phase de test initiale à 5 époques).

Taille d'image : 640px.

Optimiseur : SGD (Stochastic Gradient Descent).

Data Augmentation : Intégration de mosaïques, rotations, et ajustements de luminosité/contraste pour simuler les variations d'éclairage des magasins.

📈 Analyse des Résultats
Le notebook génère automatiquement les métriques de performance dans le dossier runs/detect/train/ :

mAP@50 : Précision moyenne à un seuil d'IoU de 0.5.

Matrice de Confusion : Analyse des Vrais Positifs (TP) et Faux Négatifs (FN).

Courbe F1-Score : Détermination du seuil de confiance optimal (identifié à 0.349 pour notre modèle).

Performances obtenues :
Validation : 87 % de détections correctes.

Test (Inédit) : 89 % de détections correctes.

Vitesse : ~118 ms par image sur GPU.

🚦 Instructions d'utilisation
Lancement : Ouvrez le fichier .ipynb dans votre environnement de notebook.

Installation : Exécutez la première cellule pour installer ultralytics.

Chemins : Modifiez la variable data_path pour pointer vers votre dossier de dataset.

Exécution : Lancez "Run All". Le modèle final sera sauvegardé sous le nom best.pt.

📦 Sorties (Outputs)
À la fin de l'exécution, le notebook exporte :

best.pt : Les poids optimisés du modèle pour déploiement dans l'application Streamlit.

results.png : Graphiques de convergence des pertes (Loss).

confusion_matrix.png : Visualisation des performances de classification.