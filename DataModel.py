# Importacion de las librerias necesarias para el desarrollo de la clase DataModel
from pydantic import BaseModel
from typing import List

class DataModel(BaseModel):

# Estas varibles permiten que la libreria pydantic haga el parseo entre el Json recibido y el modelo declarado.
# Tiene una correspondencia 1 a 1 con lo esperado.
    unnamed_0: int
    adult_mortality: float
    infant_deaths: float
    alcohol: float
    percentage_expenditure: float
    hepatitis_B: float
    measles: float
    bmi: float
    under_five_deaths: float
    polio: float
    total_expenditure: float
    diphtheria: float
    hiv_aids: float
    gdp: float
    population: float
    thinness_10_19_years: float
    thinness_5_9_years: float
    income_composition_of_resources: float
    schooling: float

#Esta funcion retorna los nombres de las columnas correspondientes con el modelo esxportado en joblib.
    def columns():
        return ["Unnamed: 0", "Adult Mortality", "infant deaths", "Alcohol","percentage expenditure","Hepatitis B", "Measles", "BMI",
                "under-five deaths", "Polio", "Total expenditure", "Diphtheria", "HIV/AIDS", "GDP", "Population",
                "thinness 10-19 years", "thinness 5-9 years", "Income composition of resources", "Schooling"]

# Clase DataModelList
class DataModelList(BaseModel):

    # Está varible permiten que la librería pydantic haga el parseo entre el Json recibido y el modelo declarado
    data: List[DataModel]

class DataEsperada(BaseModel):

# Estas varibles permiten que la librería pydantic haga el parseo entre el Json recibido y el modelo declarado.
    Life_expectancy:float

#Esta función retorna los nombres de las columnas correspondientes con el modelo esxportado en joblib.
    def columns():
        return ["Life expectancy"]

class DataEsperadaLista(BaseModel):
    dataEsperada : List[DataEsperada]
