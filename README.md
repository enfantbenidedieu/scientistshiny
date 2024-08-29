<p align="center">
	<img src="./scientistshiny.svg" height=300></img>
</p>
<div align="center">

[![GitHub](https://shields.io/badge/license-MIT-informational)](https://github.com/enfantbenidedieu/scientistshiny/blob/master/LICENSE)
[![PyPI](https://img.shields.io/pypi/v/scientistshiny.svg?color=dark-green)](https://pypi.org/project/scientistshiny/)
[![Downloads](https://static.pepy.tech/badge/scientistshiny)](https://pepy.tech/project/scientistshiny)
[![Downloads](https://static.pepy.tech/badge/scientistshiny/month)](https://pepy.tech/project/scientistshiny)
[![Downloads](https://static.pepy.tech/badge/scientistshiny/week)](https://pepy.tech/project/scientistshiny)
</div>

# scientistshiny : Perform Factorial Analysis from `scientisttools` with a Shiny for Python Application

## 1 About scientistshiny

scientistshiny is a Python package to easily improve multivariate Exploratory Data Analysis graphs.

## 2 Installation

### 2.1 Dependencies

scientistshiny requires :

```bash
scientisttools>=0.1.6
numpy>=1.26.4
matplotlib>=3.8.4
scikit-learn>=1.2.2
pandas>=2.2.3
plotnine>=0.10.1
```

### 2.2 User installation

You can install scientisttools using `pip` :

```bash
pip install scientistshiny
```

## 3 Example with `PCAshiny`

```python
# Load dataset and functions
from scientisttools import PCA, load_decathlon2
from scientistshiny import PCAshiny
decathlon = load_decathlon2()

# PCA with scientistshiny
res_shiny = PCAshiny(model = decathlon)
res_shiny.run()

# PCAshiny on a result of a PCA
res_pca = PCA(ind_sup=list(range(23,27)),quanti_sup=[10,11],quali_sup=12)
res_pca.fit(decathlon)
res_shiny = PCAshiny(model = res_pca)
res_shiny.run()
```

## 3 Author

Duvérier DJIFACK ZEBAZE ([djifacklab@gmail.com](djifacklab@gmail.com))