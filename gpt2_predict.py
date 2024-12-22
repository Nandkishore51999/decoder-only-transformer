import traceback

from flask import Flask, request, jsonify
import torch
from gpt2 import GPT, GPTConfig
import tiktoken
import torch.nn.functional as F

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print("device: ", device)
app = Flask(__name__)
enc = tiktoken.get_encoding('gpt2')

loaded_models = {}


def load_model(model_path):
    if model_path not in loaded_models:
        # print(f"Model {model_name} not already loaded...")
        # model = GPT(GPTConfig()).to(device)
        # model_path = "log_dec11_30000_Steps/"+model_name+".pt"
        # model.load_state_dict(torch.load(model_path))
        # model.eval()
        #
        # loaded_models[model_name] = model

        # model_path = "log_dec11_30000_Steps/" + model_name + ".pt"
        model = GPT(GPTConfig())
        model.to(device)
        checkpoint = torch.load(model_path)
        state_dict = checkpoint['model']
        new_state_dict = {key.replace("_orig_mod.", ""): value for key, value in state_dict.items()}

        model.load_state_dict(new_state_dict)
        model.eval()
        loaded_models[model_path] = model
    return loaded_models[model_path]


@app.route('/gpt2', methods=['POST'])
def generate_text():
    try:
        if request.method == 'POST':
            # Extract form data
            model_path = request.form['model_path']
            input_text = request.form['input_text']
            max_length = int(request.form['max_length'])
            num_return_sequences = int(request.form['num_return_sequences'])

            output = ""
            # Load model and tokenizer
            model = load_model(model_path)

            tokens = enc.encode(input_text)
            tokens = torch.tensor(tokens, dtype=torch.long)
            tokens = tokens.unsqueeze(0).repeat(num_return_sequences, 1)  # # (B, length on tokens after encoding the text)
            x = tokens.to(device)

            while x.size(1) < max_length:
                with torch.no_grad():  # telling pytorch that we will not be calling backward on any of below steps so it doesn't have to cache all the intermediate tensors
                    logits, _ = model(x)  # shape (B, T, vocab_size)
                    logits = logits[:, -1, :]  # takes only last logits which are the prediction # shpae (B, vocab_size)
                    probs = F.softmax(logits, dim=-1)
                    topk_probs, topk_indices = torch.topk(probs, 50, dim=-1)  # doing topK sampling, shape (B, 50), (B, 50), Helps in keeping the model on track (HOW???)
                    ix = torch.multinomial(topk_probs, 1)  # select a token from top-k probabilities (B, 1)
                    xcol = torch.gather(topk_indices, -1, ix)  # (B, 1)
                    x = torch.cat((x, xcol), dim=1)

            for i in range(num_return_sequences):
                tokens = x[i, :max_length].tolist()
                decode = enc.decode(tokens)
                output += f"{i}: {decode} \n=========================================================================\n"
            return output
        else:
            return "Only POST method supported"
    except Exception as e:
        traceback.print_exc()
        return e


if __name__ == '__main__':
    app.run(debug=True)

