import torch
from transformers import (
    AutoTokenizer, AutoModel, pipeline,
    AutoModelForCausalLM, BitsAndBytesConfig
)
from peft import PeftModel
from huggingface_hub import login

# Paste your API token here
api_token = ""

login(api_token)

class EmbeddingModel:
    def __init__(self, model_id="neuml/pubmedbert-base-embeddings"):
        """Initialize the embedding model with the given model ID."""
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModel.from_pretrained(model_id)
        
    def meanpooling(self, output, mask):
        """Perform mean pooling on the token embeddings.

        Args:
            output (torch.Tensor): The output tensor from the model.
            mask (torch.Tensor): The attention mask tensor.

        Returns:
            torch.Tensor: The mean pooled embeddings.
        """
        embeddings = output[0]
        mask = mask.unsqueeze(-1).expand(embeddings.size()).float()
        return torch.sum(embeddings * mask, 1) / torch.clamp(mask.sum(1), min=1e-9)

    def create_embeddings(self, text):
        """Create embeddings for the given text.

        Args:
            text (str): The input text to be embedded.

        Returns:
            torch.Tensor: The embeddings of the input text.
        """
        inputs = self.tokenizer(text, padding='max_length', truncation=True, max_length=512, return_tensors='pt').to(self.model.device)
        with torch.no_grad():
            output = self.model(**inputs)
        embeddings = self.meanpooling(output, inputs['attention_mask'])
        return embeddings
    

class LLM:
    def __init__(self, model_id="hakeematyab/CareConnect-v2-Llama-3-8B", cache_dir='hugginface_models'):
        """Initialize the language model with the given model ID and cache directory."""
        self.tokenizer = AutoTokenizer.from_pretrained(model_id, cache_dir=cache_dir)
        self.tokenizer.padding_side = 'left'
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16
        )
        self.model = AutoModelForCausalLM.from_pretrained(model_id, quantization_config=bnb_config, device_map={"": 0}, cache_dir=cache_dir)
        self.history = ['''### system: 
You are CareConnect, an expert medical personal assistant.

### instruction: 
Answer the user's queries truthfully and accurately, based on the provided context if the context is applicable. Refuse to answer any questions unrelated to medicine.
''']
        self.isMemFull = False
        
    def inference(self, query, context=""):
        """Generate a response from the model based on the query and context.

        Args:
            query (str): The user's query.
            context (str): The context to be provided to the model.

        Returns:
            str: The generated response from the model.
        """
        input_text = self.get_prompt(query, context)
        if self.isMemFull:
            if len(self.history) > 2:
                del self.history[1]
                del self.history[2]
        self.history.append(input_text)
        input_text = '\n'.join(self.history)
        inputs = self.tokenizer(input_text, return_tensors="pt", truncation=True, max_length=1024)
        if inputs['input_ids'].size(1) >= 1024:
            self.isMemFull = True
        else:
            self.isMemFull = False
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
        params = {
            "max_new_tokens": 100,
            "do_sample": True,
            "top_k": 50,
            "top_p": 0.95,
            "temperature": 0.7,
            "num_beams": 5,
            "early_stopping": True,
            "no_repeat_ngram_size": 2
        }

        outputs = self.model.generate(
            **inputs,
            **params
        )
        decoded_output = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        decoded_output = decoded_output[len(input_text):]
        self.history.append(decoded_output)
        return decoded_output
    
    def get_prompt(self, query, context):
        """Construct the input prompt for the model.

        Args:
            query (str): The user's query.
            context (str): The context to be included in the prompt.

        Returns:
            str: The formatted input prompt.
        """
        prompt_template = '''### context: 
{context}

### user: 
{user_query}

### system: 
'''
        return prompt_template.format(context=context, user_query=query)
