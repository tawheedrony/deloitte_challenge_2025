from layers.Quantum_Hybrid import build_quantum_model_class
from models.Pyraformer import Model as BaseModel

Model = build_quantum_model_class(BaseModel, "Pyraformer")
