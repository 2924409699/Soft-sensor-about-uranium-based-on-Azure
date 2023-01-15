# ---------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# ---------------------------------------------------------
import json
import logging
import os
import pickle
import numpy as np
import pandas as pd
import joblib

import azureml.automl.core
from azureml.automl.core.shared import logging_utilities, log_server
from azureml.telemetry import INSTRUMENTATION_KEY

from inference_schema.schema_decorators import input_schema, output_schema
from inference_schema.parameter_types.numpy_parameter_type import NumpyParameterType
from inference_schema.parameter_types.pandas_parameter_type import PandasParameterType
from inference_schema.parameter_types.standard_py_parameter_type import StandardPythonParameterType

data_sample = PandasParameterType(pd.DataFrame({"aluminium\u00c2\u00c1": pd.Series([0.0], dtype="float64"), "ammonia\u00b0\u00b1": pd.Series([0.0], dtype="float64"), "arsenic\u00c9\u00e9": pd.Series([0.0], dtype="float64"), "barium\u00b1\u00b5": pd.Series([0.0], dtype="float64"), "cadmium\u00ef\u00d3": pd.Series([0.0], dtype="float64"), "chloramine\u00c2\u00c8\u00b0\u00b7": pd.Series([0.0], dtype="float64"), "chromium\u00b8\u00f5": pd.Series([0.0], dtype="float64"), "copper\u00cd\u00ad": pd.Series([0.0], dtype="float64"), "flouride\u00b7\u00fa\u00bb\u00af\u00ce\u00ef": pd.Series([0.0], dtype="float64"), "bacteria\u00cf\u00b8\u00be\u00fa": pd.Series([0.0], dtype="float64"), "viruses\u00b2\u00a1\u00b6\u00be": pd.Series([0.0], dtype="float64"), "lead\u00c7\u00a6": pd.Series([0.0], dtype="float64"), "nitrates\u00cf\u00f5\u00cb\u00e1\u00d1\u00ce": pd.Series([0.0], dtype="float64"), "nitrites\u00d1\u00c7 \u00cf\u00f5\u00cb\u00e1\u00d1\u00ce": pd.Series([0.0], dtype="float64"), "mercury\u00b9\u00af": pd.Series([0.0], dtype="float64"), "perchlorate\u00b8\u00df\u00c2\u00c8\u00cb\u00e1": pd.Series([0.0], dtype="float64"), "radium\u00c0\u00d8": pd.Series([0.0], dtype="float64"), "selenium\u00ce\u00f8": pd.Series([0.0], dtype="float64"), "silver\u00d2\u00f8": pd.Series([0.0], dtype="float64")}))
input_sample = StandardPythonParameterType({'data': data_sample})

result_sample = NumpyParameterType(np.array([0.0]))
output_sample = StandardPythonParameterType({'Results':result_sample})
sample_global_parameters = StandardPythonParameterType(1.0)

try:
    log_server.enable_telemetry(INSTRUMENTATION_KEY)
    log_server.set_verbosity('INFO')
    logger = logging.getLogger('azureml.automl.core.scoring_script_v2')
except:
    pass


def init():
    global model
    # This name is model.id of model that we want to deploy deserialize the model file back
    # into a sklearn model
    model_path = os.path.join(os.getenv('AZUREML_MODEL_DIR'), 'model.pkl')
    path = os.path.normpath(model_path)
    path_split = path.split(os.sep)
    log_server.update_custom_dimensions({'model_name': path_split[-3], 'model_version': path_split[-2]})
    try:
        logger.info("Loading model from path.")
        model = joblib.load(model_path)
        logger.info("Loading successful.")
    except Exception as e:
        logging_utilities.log_traceback(e, logger)
        raise

@input_schema('Inputs', input_sample)
@input_schema('GlobalParameters', sample_global_parameters, convert_to_provided_type=False)
@output_schema(output_sample)
def run(Inputs, GlobalParameters=1.0):
    data = Inputs['data']
    result = model.predict(data)
    return {'Results':result.tolist()}
