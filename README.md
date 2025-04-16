# LLM-Filters
To run this you will need to download your desired model and follow the instructions in the code to adjust the paramaters effectively

you can this convert this to a gguf format by:
1. git clone https://github.com/ggerganov/llama.cpp
2. cd llama.cpp
3. pip install -r requirements.txt
4. python llama.cpp/convert-hf-to-gguf.py ./phi3 --outfile output_file.gguf --outtype q8_0

from here you will need to create a modelfile and than use that model file to add the model to ollama to run


datasets: 
1. https://huggingface.co/datasets/mlabonne/harmless_alpaca
2. https://huggingface.co/datasets/mlabonne/harmful_behaviors
   
