# Overview

The model combines the characteristics of URLs in the field of web attacks at the character level.It treats URLs, file paths, and registries as a short string.The model contains character-level embedding and a convolutional neural network to extract features, and the extracted features is passed onto the hidden layers for classification.

## Requirements

```
Python 3^
```

### Install dependencies:

You may use a virtual env:

```
python -m venv venv
source venv/bin/activate
```

Installing the depencencies:

```
pip install -r requirements.txt
```

## Creating and Training

```
python Detection_Model.py
```

## Evaluating

```
python evaluate.py
```

## Using predictor in code

```python
import predict

predict.predict(["posthttp://localhost:8080/tienda1/publico/vaciar.jsp?b2=vaciar+carrito';+drop+table+usuarios;+select+*+from+datos+where+nombre+like+'%"])
```


