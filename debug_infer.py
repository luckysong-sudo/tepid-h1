import torch
from tepid_h1.config import TepidH1Config
from tepid_h1.inference import InferenceEngine, GenerateConfig
from tepid_h1.modeling import TepidH1CausalLM

config = TepidH1Config.smoke()
object.__setattr__(config, 'max_position_embeddings', 256)

model = TepidH1CausalLM(config)
input_ids = torch.tensor([[1, 2, 3]])

print('=== ENGINE GENERATE max_new_tokens=2 ===')
engine = InferenceEngine(model, use_kv_cache=True)
generated, metadata = engine.generate(
    input_ids,
    config=GenerateConfig(max_new_tokens=2, do_sample=False),
)
print('SUCCESS! generated:', generated)
print('metadata:', metadata)

print('\n=== ENGINE GENERATE max_new_tokens=5 ===')
engine2 = InferenceEngine(TepidH1CausalLM(config), use_kv_cache=True)
generated2, metadata2 = engine2.generate(
    input_ids,
    config=GenerateConfig(max_new_tokens=5, do_sample=False),
)
print('SUCCESS! generated:', generated2)
print('metadata:', metadata2)

print('\n=== kwargs override ===')
engine3 = InferenceEngine(TepidH1CausalLM(config), use_kv_cache=True)
generated3, metadata3 = engine3.generate(
    input_ids,
    max_new_tokens=3,
    do_sample=False,
)
print('SUCCESS! generated:', generated3)
print('metadata:', metadata3)