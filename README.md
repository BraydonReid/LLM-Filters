# LLM-Filters
<ins>Purpose</ins> 

We built this model to help understand the fragility of LLM models and how to better defend against Multi-turn prompt injection attacks.
This repository displays the refusal directions and what it took to acheive it in order to modify a model to bypass it's preset refusal directions and display harmful responses. 
The functionality of this model does fall short when it comes to having it fully trained.

<ins>Running the Code</ins>

To run this, you will need to download your desired model and follow the instructions in the code to adjust the parameters effectively.

1st step:
Download the Large Language Model that you want to unfilter

2nd Step:
Run the training model with your model directory selected

3rd Step:
Use convert.py to convert the model back to a Hugging Face format

4th Step: Convert to a gguf format
You can convert this to a gguf format by:
1. git clone https://github.com/ggerganov/llama.cpp
2. cd llama.cpp
3. pip install -r requirements.txt
4. python convert_hf_to_gguf.py ./abliterated-deepseek-r1-7b --outfile deepseek(abliterated.gguf --outtype q8_0

5th Step:
You can then import this into Ollama by:
1. create a modelfile
2. Put the model file and gguf into the same directory
3. run: ollama create deepseek-abl
4. Then you can run the model by: ollama run deepseek-abl

6th Step:
After this, you run "run_test.py" to compare the results of the two models.


<ins> Datasets: </ins> 
1. Dataset of harmless prompts: https://huggingface.co/datasets/mlabonne/harmless_alpaca
2. Dataset of harmful prompts:  https://huggingface.co/datasets/mlabonne/harmful_behaviors

<ins>Resources Used</ins>

Uncensor any LLM with abliteration: 

   https://huggingface.co/blog/mlabonne/abliteration

Derail Yourself: Multi-turn LLM Jailbreak Attack through Self-discovered Clues: 
   
   https://paperswithcode.com/paper/derail-yourself-multi-turn-llm-jailbreak

Prompt Infection: LLM-to-LLM Prompt Injection within Multi-Agent Systems

   https://arxiv.org/abs/2410.07283?utm_source=chatgpt.com
   
