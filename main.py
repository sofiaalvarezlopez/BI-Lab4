# Importacion de las librerias necesarias para el modelo
import json
import fastapi
from typing import Optional
from sklearn.metrics import r2_score
from pandas import json_normalize
import pandas as pd
import pickle
from DataModel import DataModel, DataModelList
from DataModel import DataEsperadaLista
from clases import columnDropperTransformer, columnZeroToNaNTransformer, outOfRangeTransformer
from joblib import load

app = fastapi.FastAPI(title= "Laboratorio 4 BI", description="Realizado por Sofía Alvarez, Brenda Barahona, Alvaro Plata ", version="1.0.1")


@app.get("/")
def read_root():
   return {"Hello": "World"}


@app.get("/items/{item_id}")
def read_item(item_id: int, q: Optional[str] = None):
   return {"item_id": item_id, "q": q}

@app.post("/predict")
def make_predictions(dataModel:DataModelList):
    diccionario = fastapi.encoders.jsonable_encoder(dataModel)
    df = json_normalize(diccionario['data']) 
    df.columns = DataModel.columns()
    model = load("assets/modelo.pkl")
    print(model)
    result = model.predict(df)
    lista = result.tolist()
    json_prediccion = json.dumps(lista)
    return {"predict": json_prediccion}

@app.post("/r2")
def get_r2(data:DataModelList, dataEsperada:DataEsperadaLista):
    diccionario = fastapi.encoders.jsonable_encoder(data)
    df = json_normalize(diccionario['data']) 
    df.columns = DataModel.columns()
    model = load("assets/modelo.pkl")
    print(model)
    result = model.predict(df)
    json = fastapi.encoders.jsonable_encoder(dataEsperada)
    y = [float(d['Life_expectancy']) for d in json['dataEsperada']]
    print(result)
    print(y)
    r2 = r2_score(y, result.tolist())
    print("R2", r2)
    return {"r^2": r2}

#n la primera, se debe enviar un JSON con los predictores X de un registro de la base de datos para obtener 
# la predicción realizada por el modelo. En la segunda, se debe enviar en 
# formato JSON un conjunto de registros incluyendo predictores X y valores esperados Y, y el API debe retornar el R^2 del modelo.