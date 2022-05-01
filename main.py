from typing import Optional

from fastapi import FastAPI
import pandas as pd
from DaraModel import DataModel
from joblib import load


app = FastAPI(title= "Laboratorio 4 BI", description="Realizado por Sofía Alvarez, Brenda Barahona, Alvaro Plata ", version="1.0.1")


@app.get("/")
async def read_root():
   return {"Hello": "World"}


@app.get("/items/{item_id}")
async def read_item(item_id: int, q: Optional[str] = None):
   return {"item_id": item_id, "q": q}

@app.post("/predict")
async def make_predictions(dataModel:DataModel):
    df = pd.DataFrame(dataModel.dict(), columns=dataModel.dict().keys(), index=[0])
    df.columns = dataModel.columns()
    model = load("assets/modelo.joblib")
    result = model.predict(df)
    return result

@app.post("/r2")
async def get_r2(dataModel:DataModel):
    df = pd.DataFrame(dataModel.dict(), columns=dataModel.dict().keys(), index=[0])
    df.columns = dataModel.columns()
    model = load("assets/modelo.joblib")
    result = model.predict(df)
    return result
#n la primera, se debe enviar un JSON con los predictores X de un registro de la base de datos para obtener 
# la predicción realizada por el modelo. En la segunda, se debe enviar en 
# formato JSON un conjunto de registros incluyendo predictores X y valores esperados Y, y el API debe retornar el R^2 del modelo.