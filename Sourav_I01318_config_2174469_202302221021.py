#!/usr/bin/env python
# coding: utf-8

# ## This is your Downloaded Blueprint Notebook ##

# In[1]:


# tags to identify this iteration when submitted
# example: codex_tags = {'env': 'dev', 'region': 'USA', 'product_category': 'A'}

codex_tags = {
}

from codex_widget_factory import utils
results_json=[]


# ### Ingestion Dataset

# In[2]:


gridtableiris = """
# Below codestring is used to plot scatter plot for iris dataset. 
import plotly.express as px
import pandas as pd
import json
import plotly.io as io
from sklearn.datasets import load_iris
from sqlalchemy import create_engine
def getLogger():
    import logging
    logging.basicConfig(filename="UIACLogger.log",
                        format='%(asctime)s %(message)s',
                        filemode='a')
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    return logger
logger = getLogger()
def read_dataset():
    # Read dataset from the sklearn
    logger.info("Read dataset file from sklearn")
    try:
        dframe = load_iris(as_frame=True).data
        dframe['color']='yellow'
        dframe['color'][60:]='red'
        return dframe
    except Exception as error_msg:
        logger.info(f"Exception occured while reading the dataset"
                    f"Error Info is  {error_msg}")
def read_database_data(sql_query):
    APPLICATION_DB_HOST = "trainingserverbatch3.postgres.database.azure.com"
    APPLICATION_DB_NAME = "Training_S3_DB"
    APPLICATION_DB_USER = "Trainingadmin"
    APPLICATION_DB_PASSWORD = "p%40ssw0rd"
    try:
        connection_uri = f"postgresql://{APPLICATION_DB_USER}:{APPLICATION_DB_PASSWORD}@{APPLICATION_DB_HOST}/{APPLICATION_DB_NAME}"
        engine = create_engine(connection_uri)
        connection = engine.connect()
        dframe = pd.read_sql_query(sql_query,con=connection)
        return dframe
    except Exception as error:
        print(f"Error occured while reading data from database"
                f"using query {sql_query} and error info: {error}")
    finally:
        if connection is not None:
            connection.close()                    
def getGraph(dframe, filters):
    logger.info(
        "Preparing scatter plot json to plot iris dataset")
    for item in filters:
        if 'All' in filters[item]:
            continue
        elif isinstance(filters[item], list):
            dframe = dframe[dframe[item].isin(filters[item])]
        else:
            dframe = dframe[dframe[item] == filters[item]]
    fig = px.scatter(dframe, x='sepal length (cm)', y='sepal width (cm)', color='color')
    # fig.show()
    logger.info(
        "Successfully prepared scatter plot json to plot iris dataset")
    return io.to_json(fig)
def generate_dynamic_table(dframe, name='Sales', grid_options={"tableSize": "small", "tableMaxHeight": "80vh", "quickSearch":True}, group_headers=[], grid="auto"):
    logger.info("Generate dynamic Grid table json from dframe")
    table_dict = {}
    table_props = {}
    table_dict.update({"grid": grid, "type": "tabularForm",
                      "noGutterBottom": True, 'name': name})
    values_dict = dframe.dropna(axis=1).to_dict("records")
    table_dict.update({"value": values_dict})
    col_def_list = []
    for col in list(dframe.columns):
        col_def_dict = {}
        col_def_dict.update({"headerName": col, "field": col})
        col_def_list.append(col_def_dict)
    table_props["groupHeaders"] = group_headers
    table_props["coldef"] = col_def_list
    table_props["gridOptions"] = grid_options
    table_dict.update({"tableprops": table_props})
    logger.info("Successfully generated dynamic Grid table json from dframe")
    return table_dict
def build_grid_table_json():
    logger.info("Preparing grid table json for Product Returns Screen")
    form_config = {}
    sql_query = 'select * from i1318_iris'
    dframe = read_database_data(sql_query)
    form_config['fields'] = [generate_dynamic_table(dframe)]
    grid_table_json = {}
    grid_table_json['form_config'] = form_config
    logger.info("Successfully prepared grid table json for Product Returns Screen")
    return grid_table_json
grid_table_json = build_grid_table_json()
dynamic_outputs = json.dumps(grid_table_json)    
"""


# In[3]:


scatterPlot = """
# Below codestring is used to plot scatter plot for iris dataset. 
import plotly.express as px
import pandas as pd
import json
import plotly.io as io
from sklearn.datasets import load_iris
def getLogger():
    import logging
    logging.basicConfig(filename="UIACLogger.log",
                        format='%(asctime)s %(message)s',
                        filemode='a')
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    return logger
logger = getLogger()
def read_dataset():
    # Read dataset from the sklearn
    logger.info("Read dataset file from sklearn")
    try:
        dframe = load_iris(as_frame=True).data
        dframe['color']='yellow'
        dframe['color'][70:]='red'
        return dframe
    except Exception as error_msg:
        logger.info(f"Exception occured while reading the dataset"
                    f"Error Info is  {error_msg}")
def getGraph(dframe, filters):
    logger.info(
        "Preparing scatter plot json to plot iris dataset")
    for item in filters:
        if 'All' in filters[item]:
            continue
        elif isinstance(filters[item], list):
            dframe = dframe[dframe[item].isin(filters[item])]
        else:
            dframe = dframe[dframe[item] == filters[item]]
    fig = px.scatter(dframe, x='sepal length (cm)', y='sepal width (cm)', color='color')
    # fig.show()
    logger.info(
        "Successfully prepared scatter plot json to plot iris dataset")
    return io.to_json(fig)
#selected_filters = {"color": 'yellow'}
dframe = read_dataset()
dynamic_outputs = getGraph(dframe, selected_filters)
"""


# In[4]:




# dynamic_result = {'Iris_plot_one' : scatterPlotIris,
#                  'Iris_plot_two' : scatterPlot,} 

dynamic_result = {
    'Question one': scatterPlot,
    'Question two': gridtableiris, }

results_json.append({
    'type': 'Dynamic Plot',
    'name': 'Dynamic Plot',
    'component': 'Dynamic Plot',
    'dynamic_visual_results': dynamic_result})
   


# ### Please save and checkpoint notebook before submitting params

# In[5]:



currentNotebook = 'Sourav_I01318_config_2174469_202302221021.ipynb'

get_ipython().system('jupyter nbconvert --to script {currentNotebook} ')


# In[6]:



utils.submit_config_params(url='https://codex-api-stage.azurewebsites.net/codex-api/projects/upload-config-params/wefKiJEOQ2Jb3wJcnbWdTA', nb_name=currentNotebook, results=results_json, codex_tags=codex_tags, args={})


# In[ ]:




