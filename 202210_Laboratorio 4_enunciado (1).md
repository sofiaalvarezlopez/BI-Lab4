# Laboratorio 4 - Despliegue de modelos de ML mediate uso de API's

## Objetivos

- Reforzar el conocimiento adquirido en la construcción de pipelines.
- Exportar un modelo de machine learning utilizando joblib.
- Construir un API para montar el modelo en producción y realizar predicciones mediante peticiones HTTP.

## Herramientas

- Librerías principales de Python para procesamiento de datos y manejo de modelos como: pandas, sklearn, numpy y joblib. 
- Framework para desarrollo de API es python: [FastApi](https://fastapi.tiangolo.com/).
<br>Nota: Si ya tiene conocimiento utilizando otro framework para la construcción de APIs con Python puede hacer uso de este. 
- IDE enfocado a data science: JupyterLab en distribución de Anaconda.
- IDE enfocado a desarrollo de aplicaciones de su preferencia: Visual Studio Code, PyCharm, entre otros.
- Cliente para la realización de peticiones al API: [Postman](https://www.postman.com/).

## Enunciado
Este enunciado contiene una guía, para la realización del laboratorio, la cual se encuentra dividida en dos partes.
La primera, consiste en profundizar en la construcción de pipelines para llevar modelos de machine learnig a producción.
Por otro lado, la segunda contiene los pasos para la creación de un API que recibe las variables predictoras en formato JSON y a partir de estas 
realiza una predicción la cual será devuelta al cliente en el mismo formato. 

**Al final se espera tener el modelo del laboratorio 3 debidamente construido en un pipeline y desplegado en un API simulando un ambiente de producción. La API debe contar con dos URL habilitadas. En la primera, se debe enviar un JSON con los predictores X de un registro de la base de datos para obtener la predicción realizada por el modelo. En la segunda, se debe enviar en formato JSON un conjunto de registros incluyendo predictores X y valores esperados Y, y el API debe retornar el R^2 del modelo.**
## Construcción de Pipelines
Para realizar esta sección se recomienda utilizar JupyterLab para la construcción del Pipeline y la exportación del modelo. 

 El objetivo de crear un [pipline](https://scikit-learn.org/stable/modules/generated/sklearn.pipeline.Pipeline.html)  es automatizar todos los pasos realizados sobre los datos. Desde que salen de su fuente hasta que son ingresados al modelo de machine learning. Para un problema clásico, estos pasos incluyen la selección de features o columnas, la imputación de valores no existentes, la codificación de variables categóricas utilizando diferentes técnicas como Labael Encoding o One Hot Encoding y el escalamiento de variables numéricas en caso de ser necesario. Sin embargo, note que para problemas como el procesamiento de textos los pasos necesarios serían diferentes. Además, como último paso, el pipeline contiene el modelo que recibe los datos después de la tranformación para realizar predicciones. Finalmente, estos pipelines pueden resultar muy útiles a la hora de calibrar y comprar modelos, pues se tiene la certeza de que los datos de entrada son los mismos para todos. Incluso, pueden ser utilizados para realizar validación cruzada utilizando GridSerchCV o RandomizedSerchCV. Así mismo, pueden ser exportados para llevar los modelos a producción por medio de la serialización de estos en archivos .pkl o .joblib. 

La librería Scikit Learn cuenta con API para la creación de pipelines en la que pueden ser utilizados diferentes steps para la transformación de los datos que serán aplicados secuencialmente. Note que estos steps implementan los métodos fit y transform para ser invocados desde el pipeline. Por otro lado, los modelos que serán la parte final del proceso de automatización solo cuentan con método fit. Una vez construido el modelo es posible serializar este haciendo uso de la función dump de la librería joblib, para posteriormete deserializar, cargar (mediante la función load) y utilizar este cualquier otra aplicación o ambiente. Tenga en cuenta que la serialización de un modelo solo incluye la estructura y configuraciones realizadas sobre el pipeline, más no las instancias de los objetos que lo componen. Pues estos son provistos por la librería, por medio de la importación, en cualquiera que sea su ambiente de ejecución. Esto significa que si usted construye transformaciones personalizadas, debe incluir por separado estas en el ambiente donde cargará y ejecutará el modelo una vez sea exportado, ya que estas no están incluidas en la serialización. 

Basándose en los pasos realizados para la calibración de su modelo de regresión del laboratorio 3. Construya un pipeline que incluya todos los pasos necesarios para transformar los datos desde el archivo fuente para que estos puedan ser utilizados para realizar predicciones.

A continuación puede encontrar algunos artículos que pueden ser de utilidad para la construcción de pipelines. 
<br>
<br>
[Scikit-learn Pipeline Tutorial with Parameter Tuning and Cross-Validation](https://towardsdatascience.com/scikit-learn-pipeline-tutorial-with-parameter-tuning-and-cross-validation-e5b8280c01fb)
<br>
[Data Science Quick Tip #003: Using Scikit-Learn Pipelines!](https://towardsdatascience.com/data-science-quick-tip-003-using-scikit-learn-pipelines-66f652f26954)
<br>
[Data Science Quick Tip #004: Using Custom Transformers in Scikit-Learn Pipelines!](https://towardsdatascience.com/data-science-quick-tip-004-using-custom-transformers-in-scikit-learn-pipelines-89c28c72f22a)

## Construcción del API
Para esta sección se recomienda utilizar el IDE enfocado a desarrollo para realizar la construcción del API. Tenga en cuenta que la siguiente guía está desarrollada para el framework recomendado ([FastApi](https://fastapi.tiangolo.com/) ), sin embargo, el uso de este no es obligatorio.  

1. Crear un proyecto nuevo proyecto de Python con el nombre Lab 4 - API 
2. Instalar las dependencias necesarias para la construcción del API. 
    - Instalar el framework ingresando los siguientes comandos en la terminal. Más información en la [documentación de instalación](https://fastapi.tiangolo.com/#installation). La dependeincia uvicorn corresponde con un servido ASGI que simula el ambiente de producción. 
    ``` 
    pip install fastapi
    pip install "uvicorn[standard]" 
    ```
    
3. En un archivo main.py crear el API básico mostrado en la [documentación](https://fastapi.tiangolo.com/#installation) 
     ``` 
    from typing import Optional
    
    from fastapi import FastAPI
    
    app = FastAPI()
    
    
    @app.get("/")
    def read_root():
        return {"Hello": "World"}
    
    
    @app.get("/items/{item_id}")
    def read_item(item_id: int, q: Optional[str] = None):
        return {"item_id": item_id, "q": q}
    ``` 
    
4. Correr el servidor ingresando el siguiente comando en la terminal y verificar que el funcionamiento es correcto
     ``` 
    uvicorn main:app --reload
     ``` 
     
5. Puede verificar que el API está creado correctamente mediante la consulta de la documentación
     ```
    http://127.0.0.1:8000/docs
     ```
     
6. Crear en una archivo DaraModel.py la clase que simboliza un registro de la base de datos que se recibirá por parte del cliente. En este caso, estos coinciden con todos los datos originales en la tabla, sin contar la variable a predecir. 
La librería [pydantic](https://pydantic-docs.helpmanual.io/) realiza sugerencias de tipo, en tiempo de ejecución y provee mensajes de error amigables con el usuario cuando los datos son inválidos.

    ```
    from pydantic import BaseModel

    class DataModel(BaseModel):
    
    # Estas varibles permiten que la librería pydantic haga el parseo entre el Json recibido y el modelo declarado.
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
        income_composition_of_resources	: float
        schooling: float
    
    #Esta función retorna los nombres de las columnas correspondientes con el modelo esxportado en joblib.
        def columns(self):
            return ["Adult Mortality", "infant deaths", "Alcohol","percentage expenditure","Hepatitis B", "Measles", "BMI",
                    "under-five deaths", "Polio", "Total expenditure", "Diphtheria", "HIV/AIDS", "DGP", "Population",
                    "thinness 10-19 years", "thinness 5-9 years", "Income composition of resources", "Schooling"]

    ```

7. Crear un archivo PredictionModel.py que contiene la clase Model la cual representa el modelo que será cargado. En el constructor se crea una instacia del modelo con base en el archivo joblib. 
Adicionalmente, se crea una función para realizar las predicciones.
    ```
    from joblib import load

    class Model:
    
        def __init__(self,columns):
            self.model = load("assets/modelo.joblib")
    
        def make_predictions(self, data):
            result = self.model.predict(data)
            return result
    ```
    
8. En el archivo main.py se debe crear una función encargada de tomar los datos recibidos en el cuerpo de la petición y tranformarlos en un dataframe para que estos puedan ser recibidos por el modelo. Posteriormente, se crea una instancia del modelo y se realizan las predicciones. 
    ```
    @app.post("/predict")
    def make_predictions(dataModel: DataModel):
        df = pd.DataFrame(dataModel.dict(), columns=dataModel.dict().keys(), index=[0])
        df.columns = dataModel.columns()
        model = load("assets/modelo.joblib")
        result = model.predict(df)
        return result
    ```

9. Despliegue el API de nuevo y verifique el funcionamiento. Realice las correcciones que sean necesarias para garantizar el buen funcionamiento. 


## Entregables 
1. Jupyter notebook con la construcción del pipeline y la explicación de cada uno de los pasos realizados. 
2. Repositorio de GitHub con la API con dos URL habilitadas. En la primera, se debe enviar un JSON con los predictores X de un registro de la base de datos para obtener la predicción realizada por el modelo. En la segunda, se debe enviar en formato JSON un conjunto de registros incluyendo predictores X y valores esperados Y, y el API debe retornar el R^2 del modelo. 
3. Archivo requirements.txt con las dependencias necesarias para la ejecución del programa. 
4. Archivo README con las instrucciones de instalación, despliegue y funcionamiento del API. 
5. Documento en PDF que contenga lo siguiente: 
    - Al menos 5 escenarios de prueba del API variando los datos de entrada. Entre estos debe incluir escenarios que funcionen y que la predicción sea coherente y escenarios donde el resultado de la predicción sea incoherente o haya fallas 
    en el funcionamiento del API o Pipeline. Para cada uno de estos escenarios debe documentar los datos usados en formato JSON (adjuntos como texto plano y no como imagen) y adjuntar evidencia fotográfica de la prueba realizada en Postman que incluya el resultado obtenido. 
     Además, debe comentar brevemente si el resultado obtenido es coherente y caso de no serlo o presentarse fallas, analizar por qué se pueden estar produciendo. 
    - Párrafo corto donde se exponga una estrategia que se debe desarrollar sobre el software para mitigar incoherencias en el resultado y fallas en el sistema. 


## Bono
1. Construir transformaciones personalizadas e incluirlas en el pipeline y garantizar que el proceso completo es correcto.
2. Desplegar la API en un servidor gratuito como Heroku para que pueda prestar servicio a cualquiera haciendo uso de una URL. 
3. Implementar la estrategia para mitigación de errores identificados en los escenarios y documentados por ustedes en el documento de entrega.

## Instrucciones de Entrega
- El laboratorio se entrega en grupos de máximo 3 estudiantes
- Recuerde hacer la entrega por la sección unificada en Bloque Neón, antes del domingo 1 de mayo a las 22:00.   
  Este será el único medio por el cual se recibirán entregas.
- En la entrega indique la actividad o actividades realizada por cada uno de los miembros del grupo.

## Rúbrica de Calificación

A continuación se encuentra la rúbrica de calificación.


| Concepto | Porcentaje |
|:---:|:---:|
| Jupyter notebook con la construcción del pipeline y la explicación de cada uno de los pasos realizados  | 30% |
| Repositorio de GitHub con la API con dos URL habilitadas | 35% |
| Archivo requirements.txt  | 5% |
| Archivo README  | 10% |
| Documento en PDF | 20% |

La nota individual se calculará de acuerdo con las actividades realizadas por cada miembro del grupo.

Los bonos son de 0.33/5.0 en la nota definitiva del laboratorio.
 

