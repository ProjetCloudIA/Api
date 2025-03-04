from airflow import DAG
from airflow.operators.python_operator import PythonOperator
from airflow.utils.dates import days_ago
from datetime import datetime, timedelta

import os
import glob
import numpy as np
from skimage import io, color, transform, feature
from sklearn.svm import SVC
from sklearn.metrics import classification_report
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import mlflow
import mlflow.sklearn

# Paramètres pour le prétraitement
IMAGE_SIZE = (64, 64)
PIXELS_PER_CELL = (8, 8)
CELLS_PER_BLOCK = (2, 2)
ORIENTATIONS = 9

def extraire_hog(image):
    """
    Convertit une image en niveaux de gris, la redimensionne,
    et extrait les caractéristiques HOG.
    """
    if len(image.shape) == 3:
        image = color.rgb2gray(image)
    image_resized = transform.resize(image, IMAGE_SIZE, anti_aliasing=True)
    hog_features = feature.hog(
        image_resized,
        orientations=ORIENTATIONS,
        pixels_per_cell=PIXELS_PER_CELL,
        cells_per_block=CELLS_PER_BLOCK,
        block_norm='L2-Hys'
    )
    return hog_features

def charger_donnees(data_dir):
    """
    Parcourt les dossiers de la structure donnée et retourne
    deux tableaux : X (caractéristiques) et y (labels).
    """
    X, y = [], []
    for label in os.listdir(data_dir):
        label_path = os.path.join(data_dir, label)
        if os.path.isdir(label_path):
            count = 0
            for image_path in glob.glob(os.path.join(label_path, "*.jpg")):
                if count >= 2000:
                    break
                try:
                    image = io.imread(image_path)
                    features = extraire_hog(image)
                    X.append(features)
                    y.append(label)
                    count += 1
                except Exception as e:
                    print(f"Erreur lors du traitement de {image_path}: {e}")
    return np.array(X), np.array(y)

def entrainer_et_logger_modele(**context):
    """
    Fonction principale pour :
    1. Charger les données d'entraînement et de validation
    2. Entraîner le modèle SVM
    3. Évaluer le modèle
    4. Sauvegarder le tout avec MLflow
    """
    train_dir = os.path.join("images", "train")
    validation_dir = os.path.join("images", "validation")

    print("Chargement des données d'entraînement...")
    X_train, y_train = charger_donnees(train_dir)
    print(f"{len(X_train)} images d'entraînement chargées.")

    print("Chargement des données de validation...")
    X_val, y_val = charger_donnees(validation_dir)
    print(f"{len(X_val)} images de validation chargées.")

    # Création du pipeline
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('svm', SVC(kernel='linear', probability=True))
    ])

    print("Entraînement du modèle SVM...")
    pipeline.fit(X_train, y_train)

    print("Évaluation du modèle sur le jeu de validation...")
    y_pred = pipeline.predict(X_val)
    rapport = classification_report(y_val, y_pred)
    print(rapport)

    # Configuration de MLflow
    mlflow.set_tracking_uri("https://mlflow-cloud-g1-1d0d7b4ea267.herokuapp.com/")
    experiment = mlflow.set_experiment("facial_emotions_classification")

    os.environ['AWS_ACCESS_KEY_ID'] = ""
    os.environ['AWS_SECRET_ACCESS_KEY'] = ""

    # Loggage dans MLflow
    with mlflow.start_run(experiment_id=experiment.experiment_id):
        mlflow.sklearn.log_model(
            pipeline,
            "model",
            registered_model_name="emotion_recognition"
        )
        accuracy = np.mean(y_pred == y_val)
        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_param("data_dir", "images")
        mlflow.log_param("train_size", len(X_train))
        mlflow.log_param("val_size", len(X_val))

    print("Modèle sauvegardé dans MLflow avec succès.")

# Configuration de base du DAG
default_args = {
    'owner': 'vous',
    'depends_on_past': False,
    'start_date': days_ago(1),
    'retries': 0,
    'retry_delay': timedelta(minutes=1),
}

# Définition du DAG
with DAG(
    dag_id='facial_emotions_classification_dag',
    default_args=default_args,
    schedule_interval='@once',  
    catchup=False
) as dag:


    task_entraîner_et_logger = PythonOperator(
        task_id='entrainer_et_logger_modele',
        python_callable=entrainer_et_logger_modele,
        provide_context=True
    )

    task_entraîner_et_logger
