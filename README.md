# Laboratorio 4 - Inteligencia de Negocios 
## Sofía Álvarez, Brenda Barahona, Álvaro Plata

<h3>1. Despliegue de Modelos ML mediante uso de API's</h3>

La primera parte de este laboratorio consiste en profundizar en la construcción de pipelines para llevar modelos de machine learning a producción, en particular utilizando transformadores personalizados, como se usó en este laboratorio.

Por otro lado, la segunda consiste en la creación de un API que recibe las variables predictoras en formato JSON y a partir de estas realiza una predicción la cual será devuelta al cliente en el mismo formato. El despliegue se hace tanto local como remotamente.

<b>NOTA - BONOS: ¡¡¡Hicimos los bonos sugeridos!!!</b>

<h3>2. Organización del proyecto</h3>
En este repositorio, encuentra todos los archivos y entregables necesarios para este proyecto. A continuación, se describe su ubicación.
<ul>
  <li>Carpeta <b>Notebook</b>:</li>
  <ul>
      <li> Archivo <a href="https://github.com/sofiaalvarezlopez/BI-Lab4/blob/main/Notebook/Lab4-BI-Alvarez-Barahona-Plata.ipynb" target="_blank">Lab4-BI-Alvarez-Barahona-Plata.ipynb</a> con la pipeline construida para este laboratorio. </li>
      <li> Archivo <a href="https://github.com/sofiaalvarezlopez/BI-Lab4/blob/main/Notebook/DatosTrain.csv" target="_blank">DatosTrain.csv</a> con el cual se entrenó la pipeline del proyecto. </li>
      <li> <b>¡BONO!:</b> Archivo <a href="https://github.com/sofiaalvarezlopez/BI-Lab4/blob/main/Notebook/clases.py" target="_blank">clases.py</a> con las transformaciones customizadas que se implementaron sobre la pipeline. </li>
  </ul>
   <li>Carpeta <b>assets</b>:</li>
  <ul>
    <li> Archivo <a href="https://github.com/sofiaalvarezlopez/BI-Lab4/blob/main/assets/modelo.pkl">modelo.pkl</a>, que contiene el archivo en formato binario <code>.pickle</code>. Debido a problemas con la serialización de las clases customizadas en la pipeline, fue preferible utilizar esta librería, en lugar de <code>joblib</code>.
  </ul>
  <li>Carpeta <b>test</b>: Contiene todos los escenarios de prueba, decritos debidamente en el documento adjunto del <a href="https://github.com/sofiaalvarezlopez/BI-Lab4/blob/main/Informe%20lab4.docx"> informe</a>. </li>
  <li> Archivo <code>requirements.txt</code>, que contiene todas las dependencias asociadas a la ejecución del proyecto. </li>
  <li> Archivo <b>Informe lab4</b>, que contiene el documento del <a href="https://github.com/sofiaalvarezlopez/BI-Lab4/blob/main/Informe%20lab4.docx"> informe</a> para este laboratorio.
    
  </ul>
  
<h3>3. Instrucciones de instalación:</h3>
Para instalar este proyecto, por favor siga las instrucciones que se detallan a continuación:
  <ol>
  <li>Clonar el proyecto usando el comando <code>git clone https://github.com/sofiaalvarezlopez/BI-Lab4.git </code> .</li>
  <li>Instale las librerías y dependencias de este proyecto corriendo el comando <code>$ pip install -r requirements.txt</code>. </li>
  </ol>
  
<h3>4. Despliegue del proyecto:</h3>
El proyecto puede desplegarse de forma local o remota en AWS (<b>¡BONO!</b>) siguiendo las instrucciones detalladas a continuación:
  <ul>
    <li>
      <strong>Despliegue local:</strong>
      <ol>
        <li> En la carpeta raíz del proyecto, ejecute el comando <code>uvicorn main:app --reload</code> para correr el servicio de forma local.</li>
         <li>Es importante resaltar que para poder probar la API sin hacer uso de Postman, debe ingresarse a la dirección: <code> http://127.0.0.1:8000/docs </code>  </li>
      </ol>
    </li>
    <li>
      <strong>Despliegue remoto:</strong>
      <ol>
        <li>Abrir el navegador de preferencia y ingresar a la url <a href="http://3.228.160.169"  target="_blank">http://3.228.160.169/</a> donde se encuentra el proyecto alojado en una instacia EC2 de AWS.</li>
        <li>Es importante resaltar que para poder probar la API sin hacer uso de Postman, debe ingresarse a la dirección: <a href="http://3.228.160.169/docs" target="_blank"> http://3.228.160.169/docs </code>.  </li>
      </ol>
    </li>
  </ul>
  
<h3>5. Funcionamiento del endpoint:</h3>
Como se solicitaba, el API expone principalmente dos endpoints: <code>/predict</code> y <code>/r2</code>, los cuales sirven para obtener las predicciones del modelo y coeficiente de determinación R^2, respectivamente.
* <strong>Endpoint de predicciones:</strong> Se debe enviar un JSON con los predictores X de un registro de la base de datos para obtener la predicción realizada por el modelo.</li>
* <strong>Ejemplo de funcionamiento:</strong> En la carpeta <code>test</code> puede encontrar todos los escenarios de funcionamiento probados, los cuales están debidamente descritos en el documento adjunto a este laboratorio. No obstante, a continuación puede visualizar un ejemplo de uno de los escenarios propuestos:

```json
{
  "data": [
      { 
        "unnamed_0": 200,
        "adult_mortality": 16,
        "infant_deaths": 11,
        "alcohol": 5.88,
        "percentage_expenditure": 547.210141,
        "hepatitis_B": 98.0,
        "measles": 6071,
        "bmi": 26.8,
        "under_five_deaths": 12,
        "polio": 99.0,
        "total_expenditure": 4.11,
        "diphtheria": 99.0,
        "hiv_aids": 0.3,
        "gdp": 4212.549200,
        "population": 66881867.0,
        "thinness_10_19_years": 8.3,
        "thinness_5_9_years": 8.5,
        "income_composition_of_resources": 0.706,
        "schooling": 13.0
      },
      {
        "unnamed_0": 100,
        "adult_mortality": 393.0,
        "infant_deaths": 2,
        "alcohol": 5.01,
        "percentage_expenditure": 426.785566,
        "hepatitis_B": 94,
        "measles": 184,
        "bmi": 34.7,
        "under_five_deaths": 3,
        "polio": 96.0,
        "total_expenditure": 6.39,
        "diphtheria": 96.0,
        "hiv_aids": 9.0,
        "gdp": 5185.729845,
        "population": 1979882.0,
        "thinness_10_19_years": 8.4,
        "thinness_5_9_years": 8.2,
        "income_composition_of_resources": 0.661,
        "schooling": 12.2
      }
   ]
}
```

La respuesta de esta ejecución es:

```json

{
    "predict": "[74.94561162835979, 61.731706271446086]"
}

```
Esto significa que el primer dato (i.e. el primer país) tendrá una expectativa de vida de 74.9 años y, el segundo, 61.7 años.

* <strong>Endpoint de R^2:</strong> Se debe enviar en formato JSON un conjunto de registros incluyendo predictores X y valores esperados Y, y el API debe retornar el R^2 del modelo..</li>
* <strong>Ejemplo de funcionamiento:</strong> En la carpeta <code>test</code> puede encontrar todos los escenarios de funcionamiento probados, los cuales están debidamente descritos en el documento adjunto a este laboratorio. No obstante, a continuación puede ver un ejemplo de uno de los escenarios propuestos:
```json
{
  "data": {
    "data": [
      {
        "unnamed_0": "s",
        "adult_mortality": 16.0,
        "infant_deaths": 11.0,
        "alcohol": 5.88,
        "percentage_expenditure": 547.210141,
        "hepatitis_B": 98.0,
        "measles": 6071,
        "bmi": 26.8,
        "under_five_deaths": 12.0,
        "polio": 99.0,
        "total_expenditure": 4.11,
        "diphtheria": 99.0,
        "hiv_aids": 0.3,
        "gdp": 4212.549200,
        "population": 66881867.0,
        "thinness_10_19_years": 8.3,
        "thinness_5_9_years": 8.5,
        "income_composition_of_resources": 0.706,
        "schooling": 13
      },
      {
        "unnamed_0": 21,
        "adult_mortality": 393.0,
        "infant_deaths": 2,
        "alcohol": 5.01,
        "percentage_expenditure": 426.785566,
        "hepatitis_B": 94.0,
        "measles": 184.0,
        "bmi": 34.7,
        "under_five_deaths": 3.0,
        "polio": 96.0,
        "total_expenditure": 6.39,
        "diphtheria": 96.0,
        "hiv_aids": 9,
        "gdp": 5185.729845,
        "population": 1979882.0,
        "thinness_10_19_years": 8.4,
        "thinness_5_9_years": 8.2,
        "income_composition_of_resources": 0.661,
        "schooling": 12.2
      }
    ]
  },
  "dataEsperada": {
    "dataEsperada": [
      {
        "Life_expectancy": 73.7
      },
      {
        "Life_expectancy": 59.2
      }
    ]
  }
}
```
El resultado esperado de esta ejecución es:
```json
{
    "r^2": 0.9242702975164372
}
```
Lo cual significa que, para estos datos en particular, el ajuste del modelo es bastante bueno, tal que el modelo explica el 92% de las variaciones de los datos.

