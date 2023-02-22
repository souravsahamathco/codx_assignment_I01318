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


# ### Question one

# Create a screen named PLOT, create plot type widget and use pre-defined datasets from sklearn library.

# In[2]:


scatterPlotUIAC = """
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


# ## Question two

# Create a screen named Filter-Table and use dataset from task 1, dataset need to be read from Database not sklearn library. Display dataset using grid table and add screen filter to the screen.

# In[3]:


gridTableIrisUIAC = """
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
def get_filter_table(dframe, selected_filters):
    logger.info("Applying screen filters on the grid table dframe.")
    select_df = dframe.copy()
    for item in list(selected_filters):
        if isinstance(selected_filters[item], list):
            if 'All' not in selected_filters[item] and selected_filters[item]:
                select_df = select_df[select_df[item].isin(
                    selected_filters[item])]
        else:
            if selected_filters[item] != 'All':
                select_df = select_df[select_df[item]
                                      == selected_filters[item]]
    logger.info("Successfully applied screen filters on the grid table dframe.")
    return select_df            
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
    # selected_filters = {"target": 'setosa'}
    dframe = get_filter_table(dframe, selected_filters)
    form_config['fields'] = [generate_dynamic_table(dframe)]
    grid_table_json = {}
    grid_table_json['form_config'] = form_config
    logger.info("Successfully prepared grid table json for Product Returns Screen")
    return grid_table_json
grid_table_json = build_grid_table_json()
dynamic_outputs = json.dumps(grid_table_json)    
"""


# In[4]:


targetFilterUIAC = """
import pandas as pd
import json
from itertools import chain
from sqlalchemy import create_engine

APPLICATION_DB_HOST = "trainingserverbatch3.postgres.database.azure.com"
APPLICATION_DB_NAME = "Training_S3_DB"
APPLICATION_DB_USER = "Trainingadmin"
APPLICATION_DB_PASSWORD = "p%40ssw0rd"

def getLogger():
    import logging
    logging.basicConfig(filename="UIACLogger.log",
                        format='%(asctime)s %(message)s',
                        filemode='a')
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    return logger


logger = getLogger()


def read_database_data(sql_query, filename):
    logger.info(f"Read dataset file: {filename}")
    try:
        connection_uri = f"postgresql://{APPLICATION_DB_USER}:{APPLICATION_DB_PASSWORD}@{APPLICATION_DB_HOST}/{APPLICATION_DB_NAME}"
        engine = create_engine(connection_uri)
        connection = engine.connect()
        dframe = pd.read_sql_query(sql_query, con=connection)
        return dframe
    except Exception as error_msg:
        print(f"Error occured while reading data from database using query {query} and error info: {error_msg}")
    finally:
        if connection is not None:
            connection.close()


def get_response_filters(current_filter_params, df, default_values_selected, all_filters, multi_select_filters, extra_filters={}):
    logger.info("Preparing filter dictionary")
    # Usage
    # -----
    # >>> filter_df = pd.DataFrame(columns=[....])    # Optional operation
    # >>> filter_df = final_ADS.groupby(......)       # Optional operation
    # >>> default_values_selected = {}    # The default value to be selected for a filter, provide filter_name, filter_values
    # >>> all_option_filters = []         # Filters with an All option
    # >>> multi_select_filters = []       # Filters with an multi_select option
    # >>> more_filters = {}               # Extra filters, provide filter_names, filter_options
    # >>> final_dict_out = get_response_filters(current_filter_params, filter_df, default_values_selected, all_option_filters, multi_select_filters, more_filters)
    # >>> dynamic_outputs = json.dumps(final_dict_out)
    # Returns
    # -------
    # A dict object containing the filters JSON structure

    filters = list(df.columns)
    default_values_possible = {}
    for item in filters:
        default_possible = list(df[item].unique())
        if item in all_filters:
            default_possible = list(chain(['All'], default_possible))
        default_values_possible[item] = default_possible
    if extra_filters:
        filters.extend(list(extra_filters.keys()))
        default_values_possible.update(extra_filters)
    if current_filter_params:
        selected_filters = current_filter_params["selected"]
        # print(selected_filters)
        # current_filter = current_filter_params[selected_filters]
        # current_index = filters.index(current_filter)
        select_df = df.copy()
    final_dict = {}
    iter_value = 0
    data_values = []
    default_values = {}
    for item in filters:
        filter_dict = {}
        filter_dict["widget_filter_index"] = int(iter_value)
        filter_dict["widget_filter_function"] = False
        filter_dict["widget_filter_function_parameter"] = False
        filter_dict["widget_filter_hierarchy_key"] = False
        filter_dict["widget_filter_isall"] = True if item in all_filters else False
        filter_dict["widget_filter_multiselect"] = True if item in multi_select_filters else False
        filter_dict["widget_tag_key"] = str(item)
        filter_dict["widget_tag_label"] = str(item)
        filter_dict["widget_tag_input_type"] = "select",
        filter_dict["widget_filter_dynamic"] = True
        if current_filter_params:
            if item in df.columns:
                possible_values = list(select_df[item].unique())
                item_default_value = selected_filters[item]
                if item in all_filters:
                    possible_values = list(chain(['All'], possible_values))
                if item in multi_select_filters:
                    for value in selected_filters[item]:
                        if value not in possible_values:
                            if possible_values[0] == "All":
                                item_default_value = possible_values
                            else:
                                item_default_value = [possible_values[0]]
                else:
                    if selected_filters[item] not in possible_values:
                        item_default_value = possible_values[0]
                filter_dict["widget_tag_value"] = possible_values
                if item in multi_select_filters:
                    if 'All' not in item_default_value and selected_filters[item]:
                        select_df = select_df[select_df[item].isin(
                            item_default_value)]
                else:
                    if selected_filters[item] != 'All':
                        select_df = select_df[select_df[item]
                                              == item_default_value]
            else:
                filter_dict["widget_tag_value"] = extra_filters[item]
        else:
            filter_dict["widget_tag_value"] = default_values_possible[item]
            item_default_value = default_values_selected[item]
        data_values.append(filter_dict)
        default_values[item] = item_default_value
        iter_value = iter_value + 1
    final_dict["dataValues"] = data_values
    final_dict["defaultValues"] = default_values
    logger.info("Successfully prepared filter dictionary")
    return final_dict


def prepare_filter_json():
    logger.info(f"Preparing json for Filters in Iris Grid Table")
    # Prepare Filter json for Target in the Iris Grid Table.
    filename = "I1318_iris"
    sql_query = f"select * from {filename}"    
    dframe = read_database_data(sql_query, filename)
    dframe = dframe.groupby(['target']).sum().reset_index()
    filter_dframe = dframe[['target']]
    default_values_selected = {'target': 'setosa'}
    all_filters = []
    multi_select_filters = []
    # current_filter_params = {"selected": default_values_selected}
    final_dict_out = get_response_filters(
        current_filter_params, filter_dframe, default_values_selected, all_filters, multi_select_filters)
    logger.info(f"Successful prepared json for Filters in Iris Data Grid Table")
    return json.dumps(final_dict_out)


dynamic_outputs = prepare_filter_json()
# print(dynamic_outputs)
"""


# ## Question 4

# 4.Create a screen named Graph, create your own dataset and plot a graph, use any mathematical equation e.g., y = x^2, y = sqrt(x) for building relation between x and y values.

# In[5]:


#BEGIN CUSTOM CODE BELOW...

graphUIAC = """
import plotly.graph_objects as go
import pandas as pd
import numpy as np
import json
import plotly.io as io
def getLogger():
    import logging
    logging.basicConfig(filename="UIACLogger.log",
                        format='%(asctime)s %(message)s',
                        filemode='a')
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    return logger
logger = getLogger()
def plotMathematicalGraph():
    x = np.arange(20)
    fig = go.Figure(data=go.Scatter(x=x, y=x**2))
    # fig.show()
    logger.info(
        "Successfully prepared x=y^2")
    return io.to_json(fig)


selected_filters = {"target": 'versicolor'}

dynamic_outputs = plotMathematicalGraph()

"""
#END CUSTOM CODE


# In[6]:




# dynamic_result = {'Iris_plot_one' : scatterPlotIris,
#                  'Iris_plot_two' : scatterPlot,} 

dynamic_result = {
    'Plot': scatterPlotUIAC,
    'Filter table': gridTableIrisUIAC,
    'Graph': graphUIAC,}
dynamic_filter = {
    'targetFilter': targetFilterUIAC,
}
results_json.append({
    'type': 'Dynamic Plot',
    'name': 'Dynamic Plot',
    'component': 'Dynamic Plot',
    'dynamic_visual_results': dynamic_result,
    'dynamic_code_filters': dynamic_filter,})
   


# ### Please save and checkpoint notebook before submitting params

# In[7]:



currentNotebook = 'Sourav_I01318_config_2174469_202302221021.ipynb'

get_ipython().system('jupyter nbconvert --to script {currentNotebook} ')


# In[8]:



utils.submit_config_params(url='https://codex-api-stage.azurewebsites.net/codex-api/projects/upload-config-params/wefKiJEOQ2Jb3wJcnbWdTA', nb_name=currentNotebook, results=results_json, codex_tags=codex_tags, args={})


# In[ ]:




