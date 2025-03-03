from fastapi import FastAPI, WebSocket, WebSocketDisconnect, UploadFile, File
from mlflow.tracking import MlflowClient
import mlflow.pyfunc
from io import BytesIO
from PIL import Image
import numpy as np
import os
from skimage import color, transform, feature

os.environ = "" # AWS KEY ID
os.environ= "" #SECRET AWS

app = FastAPI()
mlflow.set_tracking_uri("https://mlflow-cloud-g1-1d0d7b4ea267.herokuapp.com/")
# Chargement du modèle depuis MLflow
client = MlflowClient()
model_info = client.get_registered_model('emotion_recognition').latest_versions[0]
model_path = model_info.source
model_mlflow = mlflow.pyfunc.load_model(model_path)

# Paramètres pour l'extraction des caractéristiques HOG
IMAGE_SIZE = (64, 64)  # même taille qu'à l'entraînement
PIXELS_PER_CELL = (8, 8)
CELLS_PER_BLOCK = (2, 2)
ORIENTATIONS = 9

def extraire_hog(image):
    """
    Convertit une image en niveaux de gris, la redimensionne,
    et extrait les caractéristiques HOG.
    """
    # Convertir en niveaux de gris
    image_gray = color.rgb2gray(image)
    # Redimensionner
    image_resized = transform.resize(image_gray, IMAGE_SIZE, anti_aliasing=True)
    # Extraire HOG
    hog_features = feature.hog(image_resized,
                               orientations=ORIENTATIONS,
                               pixels_per_cell=PIXELS_PER_CELL,
                               cells_per_block=CELLS_PER_BLOCK,
                               block_norm='L2-Hys')
    return hog_features

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """
    Endpoint qui reçoit une image, effectue le prétraitement (HOG) et retourne la prédiction du modèle.
    """
    contents = await file.read()
    image = Image.open(BytesIO(contents)).convert("RGB")
    image = np.array(image).astype("float32") / 255.0
    
    # Extraction des caractéristiques HOG
    hog_features = extraire_hog(image)
    # Le modèle s'attend à une entrée 2D de forme (1, n_features)
    input_data = np.expand_dims(hog_features, axis=0)
    
    prediction = model_mlflow.predict(input_data)
    
    if isinstance(prediction, np.ndarray):
        prediction = prediction.tolist()
    
    return {"prediction": prediction}


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """
    Endpoint WebSocket qui reçoit des images en binaire, effectue le prétraitement et renvoie la prédiction du modèle.
    """
    await websocket.accept()
    try:
        while True:
            # Réception d'une image sous forme de bytes.
            data = await websocket.receive_bytes()
            
            # Chargement de l'image avec PIL et conversion en RGB
            image = Image.open(BytesIO(data)).convert("RGB")
            # Conversion en tableau numpy et normalisation
            image_np = np.array(image).astype("float32") / 255.0

            # Extraction des caractéristiques HOG
            hog_features = extraire_hog(image_np)
            # Le modèle s'attend à une entrée 2D (n_samples, n_features)
            input_data = np.expand_dims(hog_features, axis=0)

            # Prédiction via le modèle chargé
            prediction = model_mlflow.predict(input_data)
            if isinstance(prediction, np.ndarray):
                prediction = prediction.tolist()

            # Envoi de la prédiction au client sous forme de JSON
            await websocket.send_json({"prediction": prediction})
    except WebSocketDisconnect:
        print("Client déconnecté")