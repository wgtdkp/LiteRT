import numpy as np
from ai_edge_litert.compiled_model import CompiledModel
from pathlib import Path

MODEL_PATH = "/home/dkp/models/resnet50.tflite"
OUTPUT_FOLDER = "/home/dkp/models/resnet50_inputs"

Path(OUTPUT_FOLDER).mkdir(parents=True, exist_ok=True)

compiled_model = CompiledModel.from_file(MODEL_PATH)

# The signature key of the first signature is used in this example.
signature_key = compiled_model.get_signature_by_index(0)["key"]
input_details = compiled_model.get_input_tensor_details(signature_key)

for tensor_name, detail in input_details.items():
    # Random data is used below, or you can read inputs from images here.
    # Use np.random.randint with dtype for integer types.
    data = np.random.random(detail["shape"]).astype(detail["dtype"])
    data.tofile(Path(OUTPUT_FOLDER) / f"{tensor_name}.raw")
