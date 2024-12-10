from langchain_community.llms import VLLM

model_name = "hiroki-rad/llm-jp-llm-jp-3-13b-16-ft"

llm = VLLM(model=model_name,
           quantization="awq")

