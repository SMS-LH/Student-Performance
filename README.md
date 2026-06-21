<!-- =================================================================== -->
<!--                          EN-TÊTE DU PROJET                           -->
<!-- =================================================================== -->

<h1 align="center">
  🎓 EduPredict – Student Performance
</h1>

<p align="center">
  <em>Prédire pour mieux accompagner – L'IA au service de la réussite scolaire</em>
</p>

<p align="center">
  <!-- Version Python -->
  <img src="https://img.shields.io/badge/python-3.12-2C5364?style=for-the-badge&logo=python&logoColor=white" alt="Python 3.12"/>
  &nbsp;
  <!-- Statut CI -->
  <img src="https://img.shields.io/github/actions/workflow/status/SMS-LH/Student-Performance/ci_cd.yml?style=for-the-badge&label=CI&logo=github" alt="CI Status"/>
  &nbsp;
  <!-- Licence -->
  <img src="https://img.shields.io/badge/licence-MIT-28a745?style=for-the-badge&logo=opensourceinitiative&logoColor=white" alt="Licence MIT"/>
  &nbsp;
  <!-- Déploiement -->
  <img src="https://img.shields.io/badge/d%C3%A9ploiement-Render-1e3c72?style=for-the-badge&logo=render&logoColor=white" alt="Render Deployment"/>
</p>

<p align="center">
  <a href="https://school-dashboard.onrender.com" target="_blank"><strong>🖥️ Dashboard en ligne</strong></a> &nbsp;|&nbsp;
  <a href="https://school-api.onrender.com/docs" target="_blank"><strong>📡 API Swagger</strong></a> &nbsp;|&nbsp;
  <a href="#installation"><strong>💻 Installation locale</strong></a>
</p>

<br/>

## 2. Contexte et objectifs

### Problématique

Dans un environnement éducatif de plus en plus tourné vers la donnée, il est essentiel pour les établissements scolaires de comprendre les facteurs qui influencent la performance académique des étudiants.
Les résultats peuvent dépendre de multiples aspects : socio-économiques, familiaux, comportementaux ou encore liés aux ressources disponibles.
Détecter rapidement les élèves en difficulté permet aux équipes pédagogiques d’intervenir de manière préventive et ciblée.

### Mission

L’objectif de ce projet est de construire un modèle de **machine learning supervisé** capable de prédire le **score d’examen** d’un étudiant à partir de ses caractéristiques personnelles, familiales et scolaires.
Le modèle doit être suffisamment simple pour être interprété par des non‑experts, tout en offrant des performances solides.

### Livrables

Le projet fournit un ensemble complet d’outils opérationnels :

- **Deux notebooks** détaillant l’analyse exploratoire et la comparaison des modèles.
- **Une API REST** (FastAPI) pour servir les prédictions en temps réel.
- **Un tableau de bord interactif** (Streamlit) destiné aux éducateurs, permettant :
  - La prédiction individuelle à partir d’un formulaire guidé.
  - L’analyse par lot via l’import d’un fichier CSV.
  - La visualisation des facteurs de risque et des recommandations personnalisées.
- **Un pipeline d’entraînement reproductible** et des **tests unitaires** pour garantir la fiabilité du code.
- **Un déploiement automatisé** sur Render via GitHub Actions.

### Public cible

Ce système s’adresse aux **enseignants**, **conseillers pédagogiques** et **chefs d’établissement** qui souhaitent un outil transparent d’aide à la décision pour le suivi des élèves.

## 3. Aperçu du projet

Le projet est structuré en deux services accessibles en ligne :

- **Dashboard Streamlit** : interface graphique permettant de saisir les informations d’un élève, d’importer un fichier CSV pour une analyse par lot, de consulter les facteurs de risque et d’obtenir des recommandations personnalisées.  
  Accès : https://school-dashboard-1ggu.onrender.com/

- **API FastAPI** : service REST documenté avec Swagger, offrant l’endpoint /predict pour obtenir une prédiction à partir des caractéristiques d’un étudiant.  
  Accès : https://student-api-ey7g.onrender.com/docs

Les deux services sont déployés automatiquement à chaque mise à jour du dépôt GitHub, assurant une parfaite synchronisation avec la dernière version du modèle.

## 4. Architecture du projet

```
school-student-performance/
├── README.md
├── requirements.txt
├── setup.py
├── .gitignore
├── render.yaml
├── mlflow.db
│
├── .github/
│   └── workflows/
│       └── ci_cd.yml
│
├── config/
│   └── config.yaml
│
├── data/
│   ├── raw/
│   │   └── .gitkeep
│   └── processed/
│       └── .gitkeep
│
├── notebooks/
│   ├── 01_exploratory_analysis.ipynb
│   └── 02_model_testing.ipynb
│
├── src/
│   ├── __init__.py
│   ├── utils.py
│   ├── data_preprocessing.py
│   ├── feature_engineering.py
│   ├── model_training.py
│   ├── model_evaluation.py
│   ├── pipeline.py
│   └── prediction.py
│
├── scripts/
│   ├── download_data.py
│   └── run_training.py
│
├── tests/
│   ├── conftest.py
│   ├── test_preprocessing.py
│   ├── test_feature_engineering.py
│   ├── test_models.py
│   ├── test_pipeline.py
│   └── test_api.py
│
├── api/
│   ├── __init__.py
│   ├── app.py
│   └── requirements_api.txt
│
├── dashboard/
│   ├── app.py
│   └── requirements_ui.txt
│
└── models/
    └── final_pipeline.pkl
```

### Description des dossiers principaux

- **.github/workflows** : workflow d'intégration continue (tests, linting) exécuté à chaque push.
- **config** : fichier de configuration centralisé (chemins, hyperparamètres, colonnes).
- **data** : données brutes téléchargées et données transformées après prétraitement.
- **notebooks** : analyse exploratoire détaillée (01) et comparaison des modèles avec MLflow (02).
- **src** : code source modulaire (prétraitement, feature engineering, entraînement, évaluation, pipeline complet, prédiction).
- **scripts** : scripts exécutables pour télécharger les données et lancer l'entraînement complet.
- **tests** : tests unitaires couvrant le prétraitement, le feature engineering, les modèles, le pipeline complet et l'API.
- **api** : application FastAPI exposant le modèle via un endpoint REST.
- **dashboard** : interface Streamlit pour l'utilisation du modèle par les équipes pédagogiques.
- **models** : artefacts de modèles sauvegardés (pipeline final au format joblib).

## 5. Données

### Source

Le jeu de données utilisé est **Student Performance Factors**, accessible sur Kaggle :  
https://www.kaggle.com/datasets/lainguyn123/student-performance-factors

Il contient **6 607 observations** (étudiants) décrites par **19 variables explicatives** et une variable cible.

### Dictionnaire des variables

| Variable | Type | Description |
|---|---|---|
| Hours_Studied | Numérique | Heures d'étude par semaine |
| Attendance | Numérique | Taux d'assiduité en pourcentage |
| Parental_Involvement | Catégorielle ordinale | Implication des parents (Low / Medium / High) |
| Access_to_Resources | Catégorielle ordinale | Accès aux ressources éducatives (Low / Medium / High) |
| Extracurricular_Activities | Catégorielle nominale | Participation à des activités extrascolaires (No / Yes) |
| Sleep_Hours | Numérique | Heures de sommeil par nuit |
| Previous_Scores | Numérique | Scores antérieurs |
| Motivation_Level | Catégorielle ordinale | Niveau de motivation (Low / Medium / High) |
| Internet_Access | Catégorielle nominale | Accès à Internet (No / Yes) |
| Tutoring_Sessions | Numérique | Nombre de séances de tutorat suivies |
| Family_Income | Catégorielle ordinale | Revenu familial (Low / Medium / High) |
| Teacher_Quality | Catégorielle ordinale | Qualité perçue de l'enseignant (Low / Medium / High) |
| School_Type | Catégorielle nominale | Type d'établissement (Public / Private) |
| Peer_Influence | Catégorielle ordinale | Influence des pairs (Negative / Neutral / Positive) |
| Physical_Activity | Numérique | Heures d'activité physique par semaine |
| Learning_Disabilities | Catégorielle nominale | Troubles de l'apprentissage diagnostiqués (No / Yes) |
| Parental_Education_Level | Catégorielle ordinale | Niveau d'éducation des parents (High School / College / Postgraduate) |
| Distance_from_Home | Catégorielle ordinale | Distance entre le domicile et l'école (Near / Moderate / Far) |
| Gender | Catégorielle nominale | Genre (Male / Female) |
| **Exam_Score** | **Numérique** | **Score à l'examen (cible à prédire, entre 55 et 100)** |

### Valeurs manquantes

Trois variables présentent des valeurs manquantes, en très faible proportion :

- Teacher_Quality : 78 absences (1,18 %)
- Parental_Education_Level : 90 absences (1,36 %)
- Distance_from_Home : 67 absences (1,01 %)

Ces valeurs sont imputées par le mode lors du prétraitement, avant l'encodage et la standardisation.

## 6. Résumé de l'analyse exploratoire

Le notebook **01_exploratory_analysis.ipynb** détaille l'exploration des données. Les principaux constats sont les suivants.

- Le jeu de données contient plusieurs milliers d'observations et une vingtaine de colonnes. La variable cible, Exam_Score, prend des valeurs comprises dans une plage cohérente avec des notes d'examen. Une seule valeur a été jugée aberrante et plafonnée.
- Trois variables présentent des valeurs manquantes en très faible proportion. Elles sont imputées par le mode lors du prétraitement.
- L'analyse des corrélations montre que l'assiduité et les heures d'étude sont les variables numériques les plus liées au score. À l'inverse, le sommeil et l'activité physique n'ont pas de lien linéaire notable avec la performance.
- Aucune multicolinéarité préoccupante n'a été détectée entre les variables numériques.
- Parmi les variables catégorielles, l'implication parentale, l'accès aux ressources, l'influence des pairs, le niveau d'éducation des parents et la distance au domicile sont les plus discriminantes. Le genre et le type d'école ne montrent quasiment aucune différence de score entre leurs modalités.
- Ces observations ont conduit à conserver toutes les variables pour la modélisation, en appliquant une standardisation aux numériques, un encodage ordinal pour les variables dotées d'un ordre naturel et un one‑hot encoding pour les nominales.

## 7. Modélisation

Le notebook **02_model_testing.ipynb** décrit l’ensemble de la phase de modélisation, du prétraitement à la sélection finale.

### Modèles testés

Neuf modèles de régression ont été évalués et comparés :

- Régression linéaire simple (référence)
- Régression Ridge
- Régression Lasso
- Elastic Net
- K plus proches voisins (KNN)
- Arbre de décision
- Forêt aléatoire
- XGBoost
- LightGBM

Chaque modèle a d’abord été testé avec ses paramètres par défaut, puis les plus prometteurs ont fait l’objet d’une optimisation par validation croisée (GridSearchCV).

### Modèle final retenu

Le modèle retenu est la **régression linéaire simple**.

### Justification

- Les performances obtenues sont équivalentes à celles des versions régularisées (Ridge, Lasso, Elastic Net) et supérieures à celles des modèles non linéaires testés.
- Sa simplicité garantit une interprétabilité maximale, indispensable pour une utilisation par des équipes pédagogiques.
- L’absence d’hyperparamètre à régler réduit les risques de dérive en production.
- L’écart entre les métriques d’entraînement et de test est négligeable, signe d’une très bonne généralisation.

### Performances

Sur le jeu de test, le modèle affiche une erreur absolue moyenne très faible (inférieure à un demi‑point de score) et explique une part importante de la variance du score d’examen. Ces résultats sont détaillés dans le notebook de modélisation.

### Variables les plus influentes

Les coefficients du modèle linéaire permettent d’identifier les facteurs qui pèsent le plus dans la prédiction :

1. L’assiduité
2. Les heures d’étude
3. L’influence des pairs
4. L’accès à Internet
5. L’accès aux ressources

Ces cinq variables constituent des leviers d’action concrets pour les enseignants et les familles.

## 8. Structure du code source (src/)

Le dossier **src/** contient les modules réutilisables qui implémentent les différentes étapes du pipeline de machine learning.

- **utils.py** : fonctions utilitaires (obtention de la racine du projet, chargement du fichier de configuration).
- **data_preprocessing.py** : nettoyage des données brutes (plafonnement du score d’examen).
- **feature_engineering.py** : construction du préprocesseur (imputation, encodage, standardisation).
- **model_training.py** : entraînement d’un modèle avec enregistrement des métriques et artefacts via MLflow.
- **model_evaluation.py** : calcul des métriques de régression (MAE, RMSE, MAPE, R²) sur les jeux d’entraînement et de test.
- **pipeline.py** : orchestration complète du workflow : chargement, nettoyage, transformation, entraînement et sauvegarde du modèle final.
- **prediction.py** : chargement du pipeline sauvegardé et prédiction pour un nouvel étudiant.


## 9. Installation et utilisation en local

### Prérequis

- **Python 3.12** (recommandé)
- **Git** pour cloner le dépôt
- **Conda** (optionnel) pour créer un environnement isolé

### Cloner le dépôt

Ouvrir un terminal et exécuter :
```
git clone https://github.com/SMS-LH/Student-Performance.git
cd school-student-performance
```

### Créer l’environnement (avec Conda)
```
conda create -n school_project python=3.12 -y
conda activate school_project
```

### Installer les dépendances

```
pip install -r requirements.txt
```


### Installer le package en mode développement
```
pip install -e 
```
Cela permet d’importer les modules **src/** depuis n’importe quel endroit du projet.

### Lancer les notebooks (optionnel)
```
jupyter lab
```
Ouvrir **notebooks/01_exploratory_analysis.ipynb** pour l’analyse exploratoire, puis **notebooks/02_model_testing.ipynb** pour la modélisation.

### Lancer l’API FastAPI
```
uvicorn api.app:app --reload
```

L’API est accessible à l’adresse **http://127.0.0.1:8000**. La documentation Swagger est disponible à **http://127.0.0.1:8000/docs**.

### Lancer le dashboard Streamlit

Dans un second terminal (avec l’environnement activé) :
```
streamlit run dashboard/app.py
```
Le dashboard est accessible à l’adresse **http://localhost:8501**.

## 10. Déploiement

Le projet est déployé sur **Render**, une plateforme cloud qui permet de mettre en ligne des services web directement depuis un dépôt GitHub.

### Services déployés

- **API FastAPI** : accessible à l'adresse https://student-api-ey7g.onrender.com/
- **Dashboard Streamlit** : accessible à l'adresse https://school-dashboard-1ggu.onrender.com/

### Fichier de configuration

Le fichier **render.yaml** présent à la racine du projet décrit les deux services web :

- Le service API, construit avec Python et exécuté via Uvicorn.
- Le service Dashboard, construit avec Python et exécuté via Streamlit.

Chaque service spécifie sa commande de build (installation des dépendances) et sa commande de démarrage.

### Déploiement automatique

Le déploiement est déclenché automatiquement à chaque push sur la branche principale du dépôt GitHub. Render détecte les modifications, reconstruit les services et les redémarre sans intervention manuelle.

### Variables d'environnement

Le dashboard utilise une variable d'environnement **API_URL** qui pointe vers l'URL de l'API déployée, afin d'effectuer les prédictions à distance.


## 11. Tests et CI/CD

### Tests unitaires

Le dossier **tests/** contient une suite de tests couvrant les principaux modules du projet :

- Prétraitement des données (plafonnement du score)
- Feature engineering (préprocesseur, imputation, encodage)
- Entraînement d’un modèle
- Pipeline complet (de la donnée brute à la sauvegarde du modèle)
- API (endpoints, validation des données)

Pour exécuter les tests :
```
pytest tests/ -v
```
La couverture de code est également disponible via l’option `--cov`.

### Intégration continue

Un workflow GitHub Actions est défini dans le fichier **.github/workflows/ci_cd.yml**.  
Il se déclenche à chaque push ou pull request sur la branche principale et exécute automatiquement deux jobs :

- **Tests** : installation des dépendances, exécution de la suite de tests avec pytest et rapport de couverture.
- **Linting** : vérification du style de code avec flake8 sur les dossiers `src/`, `tests/` et `api/`.

Ce workflow garantit que chaque modification du code est validée avant d’être fusionnée ou déployée.

## 12. Auteurs et remerciements

### Auteurs

Ce projet a été réalisé par :

- **Ndeye Khoudia DIOP** — Élève ingénieure statisticienne économiste, ISEP2, ENSAE
- **Mourtalla SALL** — Élève ingénieur statisticien économiste, ISEP2, ENSAE

### Encadrement

Ce projet s’inscrit dans le cadre du cours de **Machine Learning** dispensé par **Mme Mously DIAW**, Data Scientist, à l’École Nationale de la Statistique et de l’Analyse Économique (ENSAE).

### Remerciements

Nous remercions Mme Diaw pour son accompagnement tout au long de cette formation, ainsi que l’ensemble de l’équipe pédagogique de l’ENSAE pour les compétences transmises durant le cursus.