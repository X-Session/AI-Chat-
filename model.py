import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer

# Load the PyTorch LLM model from a .pth file
model_path = "/mnt/a/Download/consolidated.00.pth" # update this to the correct path
model = GPT2LMHeadModel.from_pretrained(model_path)

# Load the tokenizer for the LLM model
tokenizer = GPT2Tokenizer.from_pretrained(model_path)

# Set the model to evaluation mode
model.eval()

# Prompt the user for input
prompt = "How are you?"

# Encode the user's input using the tokenizer
input_ids = tokenizer.encode(prompt, return_tensors='pt')

# Generate a response from the model
with torch.no_grad():
    output = model.generate(input_ids, max_length=100, do_sample=True)

# Decode the model's response and print it to the console
response = tokenizer.decode(output[0], skip_special_tokens=True)
print(response)
