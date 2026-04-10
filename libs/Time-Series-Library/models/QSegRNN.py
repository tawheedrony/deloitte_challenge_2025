from layers.Quantum_Hybrid import build_quantum_model_class
from models.SegRNN import Model as BaseModel

Model = build_quantum_model_class(BaseModel, "SegRNN")
