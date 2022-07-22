#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
from scipy import stats
import seaborn as sns
import statsmodels.api as sm           
import statsmodels.formula.api as smf 
import multiprocessing
import random
from itertools import product


# In[2]:


import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns
plt.style.use("bmh")                    
sns.set_style({"axes.grid":False}) 
get_ipython().run_line_magic('pip', 'install plotly')
import plotly.offline as py
from plotly.offline import iplot, init_notebook_mode
import plotly.graph_objs as go
init_notebook_mode(connected = True) 


# In[3]:


import warnings as wrn
wrn.filterwarnings("ignore", category = DeprecationWarning) 
wrn.filterwarnings("ignore", category = FutureWarning) 
wrn.filterwarnings("ignore", category = UserWarning) 


# In[4]:



def bar_plot(x, y, title, yaxis, c_scale):
    trace = go.Bar(
    x = x,
    y = y,
    marker = dict(color = y, colorscale = c_scale))
    layout = go.Layout(hovermode= "closest", title = title, yaxis = dict(title = yaxis))
    fig = go.Figure(data = [trace], layout = layout)
    return iplot(fig)


def scatter_plot(x, y, title, xaxis, yaxis, size, c_scale):
    trace = go.Scatter(
    x = x,
    y = y,
    mode = "markers",
    marker = dict(color = y, size = size, showscale = True, colorscale = c_scale))
    layout = go.Layout(hovermode= "closest", title = title, xaxis = dict(title = xaxis), yaxis = dict(title = yaxis))
    fig = go.Figure(data = [trace], layout = layout)
    return iplot(fig)    
    

def plot_histogram(x, title, yaxis, color):
    trace = go.Histogram(x = x,
                        marker = dict(color = color))
    layout = go.Layout(hovermode= "closest" , title = title, yaxis = dict(title = yaxis))
    fig = go.Figure(data = [trace], layout = layout)
    return iplot(fig)


# In[5]:


url = "https://raw.githubusercontent.com/JoaquinAmatRodrigo/Estadistica-machine-learning-python/master/data/SaratogaHouses.csv"
datos = pd.read_csv(url, sep = ",")


# In[6]:


datos.head(3)


# In[7]:


datos.tail(3)


# In[8]:


datos.shape


# In[9]:


datos.info()


# In[10]:


datos.columns.values


# In[11]:


datos.dtypes.value_counts()


# In[12]:


datos.isnull().any().values


# In[13]:


valores_categoricos = datos.select_dtypes(include = ["object"])
valores_categoricos.columns.values


# In[14]:


# Variables Cualitativas o categoricas
datos.select_dtypes(include=["object"]).describe()


# In[15]:


fig, axes = plt.subplots(nrows=3, ncols=2, figsize=(9, 5))
axes = axes.flat
valores_categoricos = datos.select_dtypes(include=["object"]).columns

for i, colum in enumerate(valores_categoricos):
    datos[colum].value_counts().plot.barh(ax = axes[i])
    axes[i].set_title(colum, fontsize = 7, fontweight = "bold")
    axes[i].tick_params(labelsize = 6)
    axes[i].set_xlabel("")

fig.tight_layout()
plt.subplots_adjust(top=0.9)
fig.suptitle("Distribución de las variables cualitativas",
             fontsize = 10, fontweight = "bold");


# In[16]:


fig, axes = plt.subplots(nrows=3, ncols=2, figsize=(9, 5))
axes = axes.flat
valores_categoricos = datos.select_dtypes(include=["object"]).columns

for i, colum in enumerate(valores_categoricos):
    sns.violinplot(
        x     = colum,
        y     = "price",
        data  = datos,
        color = "white",
        ax    = axes[i]
    )
    axes[i].set_title(f"Precio vs {colum}", fontsize = 7, fontweight = "bold")
    axes[i].yaxis.set_major_formatter(ticker.EngFormatter())
    axes[i].tick_params(labelsize = 6)
    axes[i].set_xlabel("")
    axes[i].set_ylabel("")
    
fig.tight_layout()
plt.subplots_adjust(top=0.9)
fig.suptitle("Distribucion de las variable categóricas y el precio", fontsize = 10, fontweight = "bold");


# In[17]:


sns_plot = sns.distplot(datos["price"])


# In[18]:


print("Skewness: %f" %datos["price"].skew())
print("Kurtosis: %f" %datos["price"].kurt())


# In[19]:


valores_numericos = datos.select_dtypes(include = ["int64", "float64"])
valores_numericos.columns.values


# In[20]:


valores_numericos.head(3)


# In[21]:


valores_numericos.describe()


# In[22]:


fig, axes = plt.subplots(nrows=3, ncols=3, figsize=(9, 5))
axes = axes.flat
valores_numericos = datos.select_dtypes(include=["float64", "int"]).columns
valores_numericos = valores_numericos.drop('price')

for i, colum in enumerate(valores_numericos):
    sns.histplot(
        data    = datos,
        x       = colum,
        stat    = "count",
        kde     = True,
        color   = (list(plt.rcParams["axes.prop_cycle"])*2)[i]["color"],
        line_kws= {"linewidth": 2},
        alpha   = 0.3,
        ax      = axes[i]
    )
    axes[i].set_title(colum, fontsize = 7, fontweight = "bold")
    axes[i].tick_params(labelsize = 6)
    axes[i].set_xlabel("")


fig.tight_layout()
plt.subplots_adjust(top = 0.9)
fig.suptitle("Distribución de las variables numéricas", fontsize = 10, fontweight = "bold");


# In[23]:


fig, axes = plt.subplots(nrows=3, ncols=3, figsize=(9, 5))
axes = axes.flat
valores_numericos = datos.select_dtypes(include=["float64", "int"]).columns
valores_numericos = valores_numericos.drop("price")

for i, colum in enumerate(valores_numericos):
    sns.regplot(
        x           = datos[colum],
        y           = datos["price"],
        color       = "blue",
        marker      = '.',
        scatter_kws = {"alpha":0.4},
        line_kws    = {"color":"y","alpha":0.9},
        ax          = axes[i]
    )
    axes[i].set_title(f"price vs {colum}", fontsize = 7, fontweight = "bold")
    axes[i].yaxis.set_major_formatter(ticker.EngFormatter())
    axes[i].xaxis.set_major_formatter(ticker.EngFormatter())
    axes[i].tick_params(labelsize = 6)
    axes[i].set_xlabel("")
    axes[i].set_ylabel("")

    

fig.tight_layout()
plt.subplots_adjust(top=0.9)
fig.suptitle("Relación de las variables numéricas con precio", fontsize = 10, fontweight = "bold");


# In[24]:


g = sns.pairplot(datos)
plt.title("Pairplots de todas las variables")
g.map_upper(sns.kdeplot, levels=4, color=".2")
plt.show()


# In[25]:


corr_matrix = datos.corr()


# In[26]:


corr_matrix["price"].sort_values(ascending=False)


# In[27]:


plt.figure(figsize=(10,6))
sns.heatmap(datos.corr(),cmap=plt.cm.Blues,annot=True)
plt.title("Mapa de calor de las correlación entre todas las variables numéricas",
         fontsize=13)
plt.show()


# In[28]:


fig, ax = plt.subplots(10, 2, figsize = (15, 13))
sns.boxplot(x= datos["price"], ax = ax[0,0])
sns.distplot(datos["price"], ax = ax[0,1])
sns.boxplot(x= datos["lotSize"], ax = ax[1,0])
sns.distplot(datos["lotSize"], ax = ax[1,1])
sns.boxplot(x= datos["age"], ax = ax[2,0])
sns.distplot(datos["age"], ax = ax[2,1])
sns.boxplot(x= datos["landValue"], ax = ax[3,0])
sns.distplot(datos["landValue"], ax = ax[3,1])
sns.boxplot(x= datos["livingArea"], ax = ax[4,0])
sns.distplot(datos["livingArea"], ax = ax[4,1])
sns.boxplot(x= datos["pctCollege"], ax = ax[5,0])
sns.distplot(datos["pctCollege"], ax = ax[5,1])
sns.boxplot(x= datos["bedrooms"], ax = ax[6,0])
sns.distplot(datos["bedrooms"], ax = ax[6,1])
sns.boxplot(x= datos["fireplaces"], ax = ax[7,0])
sns.distplot(datos["fireplaces"], ax = ax[7,1])
sns.boxplot(x= datos["bathrooms"], ax = ax[8,0])
sns.distplot(datos["bathrooms"], ax = ax[8,1])
sns.boxplot(x= datos["rooms"], ax = ax[9,0])
sns.distplot(datos["rooms"], ax = ax[9,1])

plt.tight_layout()


# In[29]:



fig, ax = plt.subplots(figsize=(6, 3.84))

datos.plot(
    x    = "price",
    y    = "price",
    c    = "firebrick",
    kind = "scatter",
    ax   = ax
)
ax.set_title("Distribución precio");


fig, ax = plt.subplots(figsize=(6, 3.84))

datos.plot(
    x    = "lotSize",
    y    = "price",
    c    = "firebrick",
    kind = "scatter",
    ax   = ax
)
ax.set_title("Distribución de área y precio");


fig, ax = plt.subplots(figsize=(6, 3.84))

datos.plot(
    x    = "age",
    y    = "price",
    c    = "firebrick",
    kind = "scatter",
    ax   = ax
)
ax.set_title("Distribución de antigüedad y precio");


fig, ax = plt.subplots(figsize=(6, 3.84))

datos.plot(
    x    = "landValue",
    y    = "price",
    c    = "firebrick",
    kind = "scatter",
    ax   = ax
)
ax.set_title("Distribución de valor del suelo y precio");


fig, ax = plt.subplots(figsize=(6, 3.84))

datos.plot(
    x    = "livingArea",
    y    = "price",
    c    = "firebrick",
    kind = "scatter",
    ax   = ax
)
ax.set_title("Distribución de espacio vivienda y precio");


# In[30]:


datos.drop(datos[datos.price>350000].index, inplace = True)
datos.reset_index(drop = True, inplace = True)

datos.drop(datos[datos.lotSize>1].index, inplace = True)
datos.reset_index(drop = True, inplace = True)

datos.drop(datos[datos.age>50].index, inplace = True)
datos.reset_index(drop = True, inplace = True)

datos.drop(datos[datos.landValue>55000].index, inplace = True)
datos.reset_index(drop = True, inplace = True)

datos.drop(datos[datos.livingArea>3100].index, inplace = True)
datos.reset_index(drop = True, inplace = True)


# In[31]:



fig, ax = plt.subplots(figsize=(6, 3.84))

datos.plot(
    x    = "price",
    y    = "price",
    c    = "firebrick",
    kind = "scatter",
    ax   = ax
)
ax.set_title("Distribución precio");


fig, ax = plt.subplots(figsize=(6, 3.84))

datos.plot(
    x    = "lotSize",
    y    = "price",
    c    = "firebrick",
    kind = "scatter",
    ax   = ax
)
ax.set_title("Distribución de área y precio");


fig, ax = plt.subplots(figsize=(6, 3.84))

datos.plot(
    x    = "age",
    y    = "price",
    c    = "firebrick",
    kind = "scatter",
    ax   = ax
)
ax.set_title("Distribución de antigüedad y precio");


fig, ax = plt.subplots(figsize=(6, 3.84))

datos.plot(
    x    = "landValue",
    y    = "price",
    c    = "firebrick",
    kind = "scatter",
    ax   = ax
)
ax.set_title("Distribución de valor del suelo y precio");


fig, ax = plt.subplots(figsize=(6, 3.84))

datos.plot(
    x    = "livingArea",
    y    = "price",
    c    = "firebrick",
    kind = "scatter",
    ax   = ax
)
ax.set_title("Distribución de espacio vivienda y precio");


# In[32]:


fig, ax = plt.subplots(10, 2, figsize = (15, 13))
sns.boxplot(x= datos["price"], ax = ax[0,0])
sns.distplot(datos["price"], ax = ax[0,1])
sns.boxplot(x= datos["lotSize"], ax = ax[1,0])
sns.distplot(datos["lotSize"], ax = ax[1,1])
sns.boxplot(x= datos["age"], ax = ax[2,0])
sns.distplot(datos["age"], ax = ax[2,1])
sns.boxplot(x= datos["landValue"], ax = ax[3,0])
sns.distplot(datos["landValue"], ax = ax[3,1])
sns.boxplot(x= datos["livingArea"], ax = ax[4,0])
sns.distplot(datos["livingArea"], ax = ax[4,1])
sns.boxplot(x= datos["pctCollege"], ax = ax[5,0])
sns.distplot(datos["pctCollege"], ax = ax[5,1])
sns.boxplot(x= datos["bedrooms"], ax = ax[6,0])
sns.distplot(datos["bedrooms"], ax = ax[6,1])
sns.boxplot(x= datos["fireplaces"], ax = ax[7,0])
sns.distplot(datos["fireplaces"], ax = ax[7,1])
sns.boxplot(x= datos["bathrooms"], ax = ax[8,0])
sns.distplot(datos["bathrooms"], ax = ax[8,1])
sns.boxplot(x= datos["rooms"], ax = ax[9,0])
sns.distplot(datos["rooms"], ax = ax[9,1])

plt.tight_layout()


# In[33]:


datos.shape


# In[34]:


from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
                                        datos.drop("price", axis = "columns"),
                                        datos["price"],
                                        train_size   = 0.8,
                                        random_state = 1234,
                                        shuffle      = True
                                    )


# In[35]:


print(y_train.describe())


# In[36]:


print(y_test.describe())


# In[37]:


print(X_train.describe())


# In[38]:


print(X_test.describe())


# In[39]:


# Valores nulos los veremos en la parte de preprocesamiento de los datos 
X_train.isna().sum().sort_values()


# In[40]:


from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.compose import make_column_selector

numeric_cols = X_train.select_dtypes(include=["float64", "int"]).columns.to_list()
cat_cols = X_train.select_dtypes(include=["object", "category"]).columns.to_list()

# Para las variables numéricas
numeric_transformer = Pipeline(
                        steps=[
                            ("scaler", StandardScaler())]
                      )


# Para las variables categóricas
categorical_transformer = Pipeline(
                            steps=[
                                ("onehot", OneHotEncoder(handle_unknown="ignore"))
                            ]
                          )

preprocessor = ColumnTransformer(
                    transformers=[
                        ("numeric", numeric_transformer, numeric_cols),
                        ("cat", categorical_transformer, cat_cols)
                    ],
                    remainder='passthrough'
                )


# In[41]:


X_train_prep = preprocessor.fit_transform(X_train)
X_test_prep  = preprocessor.transform(X_test)


# In[42]:


encoded_cat = preprocessor.named_transformers_["cat"]["onehot"].get_feature_names(cat_cols)
labels = np.concatenate([numeric_cols, encoded_cat])
datos_train_prep = preprocessor.transform(X_train)
datos_train_prep = pd.DataFrame(datos_train_prep, columns=labels)
datos_train_prep.info()


# In[43]:


sns_plot = sns.distplot(datos_train_prep["age"])
sns_plot = sns.distplot(datos["age"])


# In[44]:


sns_plot = sns.distplot(datos_train_prep["lotSize"])
sns_plot = sns.distplot(datos["lotSize"])


# In[45]:



corr_num = datos_train_prep.loc[:,["lotSize", "age", "landValue", "rooms", "newConstruction_Yes", "centralAir_Yes"]]
corr_num = corr_num.iloc[:,]
for i in corr_num.columns:
    x = corr_num[i]
    y = y_train
    slope, intercept, r_value, p_value, std_err = stats.linregress(x,y)
    line = slope*x + intercept

   
    trace0 = go.Scatter(
                  x = x,
                  y = y,
                  mode = "markers",
                  marker = dict(color = "red"),
                  name ="Data"
                  )
    
    
    trace1 = go.Scatter(
                  x = x,
                  y = line,
                  mode="lines",
                  marker = dict(color = "black"),
                  name="Fit"
                  )

    
    title = "{} vs Precio (r: {:0.4f}, p: {})".format(corr_num[i].name, r_value, p_value)
    layout = go.Layout(
            title = title, yaxis = dict(title = "Precio"))

    data = [trace0, trace1]
    fig = go.Figure(data = data, layout = layout)
    iplot(fig)


# In[46]:


datos_train_prep.head(3)


# In[47]:


datos_train_prep.shape


# In[48]:


y_train.shape


# In[49]:


from sklearn.linear_model import Ridge
from sklearn.linear_model import RidgeCV

pipe = Pipeline([("preprocessing", preprocessor),
                 ("modelo", Ridge())])

_ = pipe.fit(X=X_train, y=y_train)


# In[50]:



from sklearn.model_selection import cross_val_score

cv_scores = cross_val_score(
                estimator = pipe,
                X         = X_train,
                y         = y_train,
                scoring   = "neg_root_mean_squared_error",
                cv        = 5
             )

print(f"Métricas validación cruzada: {cv_scores}")
print(f"Média métricas de validación cruzada: {cv_scores.mean()}")


# In[51]:


predicciones = pipe.predict(X_test)


# In[52]:


# predicciones y el valor real
df_predicciones = pd.DataFrame({"precio" : y_test, "prediccion" : predicciones})
df_predicciones.head()


# In[53]:


from sklearn.metrics import mean_squared_error

rmse = mean_squared_error(
        y_true = y_test,
        y_pred = predicciones,
        squared = False
       )
rmse


# In[54]:


from sklearn.model_selection import RandomizedSearchCV, RepeatedKFold

param_distributions = {"modelo__alpha": np.logspace(-5, 5, 500)}

grid = RandomizedSearchCV(
        estimator  = pipe,
        param_distributions = param_distributions,
        n_iter     = 20,
        scoring    = "neg_root_mean_squared_error",
        n_jobs     = multiprocessing.cpu_count() - 1,
        cv         = RepeatedKFold(n_splits = 5, n_repeats = 3), 
        refit      = True, 
        verbose    = 0,
        random_state = 123,
        return_train_score = True
       )

grid.fit(X = X_train, y = y_train)


resultados = pd.DataFrame(grid.cv_results_)
resultados.filter(regex = "(param.*|mean_t|std_t)")    .drop(columns = "params")    .sort_values("mean_test_score", ascending = False)    .head(1)


# In[55]:


modelo_final = grid.best_estimator_
predicciones = modelo_final.predict(X = X_test)
rmse_lm = mean_squared_error(
            y_true  = y_test,
            y_pred  = predicciones,
            squared = False
          )
print(f"El error (rmse) de test es: {rmse_lm}")


# In[56]:


from sklearn.neighbors import KNeighborsRegressor


pipe = Pipeline([("preprocessing", preprocessor),
                 ("modelo", KNeighborsRegressor())])


param_distributions = {"modelo__n_neighbors": np.linspace(1, 100, 500, dtype=int)}


grid = RandomizedSearchCV(
        estimator  = pipe,
        param_distributions = param_distributions,
        n_iter     = 20,
        scoring    = "neg_root_mean_squared_error",
        n_jobs     = multiprocessing.cpu_count() - 1,
        cv         = RepeatedKFold(n_splits = 5, n_repeats = 3), 
        refit      = True, 
        verbose    = 0,
        random_state = 123,
        return_train_score = True
       )

grid.fit(X = X_train, y = y_train)

resultados = pd.DataFrame(grid.cv_results_)
resultados.filter(regex = "(param.*|mean_t|std_t)")    .drop(columns = "params")    .sort_values("mean_test_score", ascending = False)    .head(1)


# In[57]:


modelo_final = grid.best_estimator_
predicciones = modelo_final.predict(X = X_test)
rmse_knn = mean_squared_error(
            y_true  = y_test,
            y_pred  = predicciones,
            squared = False
           )
print(f"El error (rmse) de test es: {rmse_knn}")


# In[58]:


from sklearn.ensemble import RandomForestRegressor

pipe = Pipeline([("preprocessing", preprocessor),
                 ("modelo", RandomForestRegressor())])

param_distributions = {
    "modelo__n_estimators": [50, 100, 1000, 2000],
    "modelo__max_features": ["auto", 3, 5, 7],
    "modelo__max_depth"   : [None, 3, 5, 10, 20]
}


grid = RandomizedSearchCV(
        estimator  = pipe,
        param_distributions = param_distributions,
        n_iter     = 20,
        scoring    = "neg_root_mean_squared_error",
        n_jobs     = multiprocessing.cpu_count() - 1,
        cv         = RepeatedKFold(n_splits = 5, n_repeats = 3),
        refit      = True, 
        verbose    = 0,
        random_state = 123,
        return_train_score = True
       )

grid.fit(X = X_train, y = y_train)


resultados = pd.DataFrame(grid.cv_results_)
resultados.filter(regex = "(param.*|mean_t|std_t)")    .drop(columns = "params")    .sort_values("mean_test_score", ascending = False)    .head(1)


# In[59]:


modelo_final = grid.best_estimator_
predicciones = modelo_final.predict(X = X_test)
rmse_rf = mean_squared_error(
            y_true  = y_test,
            y_pred  = predicciones,
            squared = False
          )
print(f"El error (rmse) de test es: {rmse_rf}")


# In[60]:


from sklearn.ensemble import GradientBoostingRegressor


pipe = Pipeline([("preprocessing", preprocessor),
                 ("modelo", GradientBoostingRegressor())])


param_distributions = {
    "modelo__n_estimators": [50, 100, 1000, 2000],
    "modelo__max_features": ["auto", 3, 5, 7],
    "modelo__max_depth"   : [None, 3, 5, 10, 20],
    "modelo__subsample"   : [0.5,0.7, 1]
}

# Búsqueda random grid
grid = RandomizedSearchCV(
        estimator  = pipe,
        param_distributions = param_distributions,
        n_iter     = 20,
        scoring    = "neg_root_mean_squared_error",
        n_jobs     = multiprocessing.cpu_count() - 1,
        cv         = RepeatedKFold(n_splits = 5, n_repeats = 3),
        refit      = True, 
        verbose    = 0,
        random_state = 123,
        return_train_score = True
       )

grid.fit(X = X_train, y = y_train)


resultados = pd.DataFrame(grid.cv_results_)
resultados.filter(regex = "(param.*|mean_t|std_t)")    .drop(columns = "params")    .sort_values("mean_test_score", ascending = False)    .head(1)


# In[61]:


modelo_final = grid.best_estimator_
predicciones = modelo_final.predict(X = X_test)
rmse_gbm = mean_squared_error(
            y_true  = y_test,
            y_pred  = predicciones,
            squared = False
          )
print(f"El error (rmse) de test es: {rmse_gbm}")


# In[62]:


from sklearn.ensemble import StackingRegressor


pipe_ridge = Pipeline([("preprocessing", preprocessor),
                     ("ridge", Ridge(alpha=3.4))])

pipe_rf = Pipeline([("preprocessing", preprocessor),
                     ("random_forest", RandomForestRegressor(
                                         n_estimators = 1000,
                                         max_features = 7,
                                         max_depth    = 20
                                        )
                     )])


# In[63]:


estimators = [("ridge", pipe_ridge),
              ("random_forest", pipe_rf)]

stacking_regressor = StackingRegressor(estimators=estimators,
                                       final_estimator=RidgeCV())

_ = stacking_regressor.fit(X = X_train, y = y_train)


# In[64]:


modelo_final = stacking_regressor
predicciones = modelo_final.predict(X = X_test)
rmse_stacking = mean_squared_error(
                    y_true  = y_test,
                    y_pred  = predicciones,
                    squared = False
                  )
print(f"El error (rmse) de test es: {rmse_stacking}")


# In[65]:


error_modelos = pd.DataFrame({
                        "modelo": ["lm", "random forest", "gradient boosting", "Knn", "Stacking"],
                        "rmse": [rmse_lm, rmse_rf, rmse_gbm, rmse_knn, rmse_stacking]
                     })
error_modelos = error_modelos.sort_values('rmse', ascending=False)

fig, ax = plt.subplots(figsize=(6, 3.84))
ax.hlines(error_modelos.modelo, xmin=0, xmax=error_modelos.rmse)
ax.plot(error_modelos.rmse, error_modelos.modelo, "o", color="black")
ax.tick_params(axis='y', which="major", labelsize=12)
ax.set_title("Comparación de error de test modelos"),
ax.set_xlabel("Test rmse");

