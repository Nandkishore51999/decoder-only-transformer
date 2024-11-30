import torch
from flask import Flask, request, jsonify
import torch
import torch.nn as nn
import os
from text_preprocessing import *
from decoder_only_transformer import DecoderOnlyTransformer

app = Flask(__name__)


def load_model(model_path, device="cuda"):
    model = DecoderOnlyTransformer.load_from_checkpoint(
        model_path,
        # 'C:\\Workspace-ML\\lightning_logs\\version_52\\checkpoints\\epoch=119-step=633960.ckpt',
        n_tokens=2685,
        d_model=64,
        max_len=1024).to(device)
    # model = torch.load(model_path, map_location=device)
    model.eval()
    return model

# C:\\Workspace-ML\\decoder-only-transformer\\lightning_logs\\version_2\\checkpoints\\epoch=999-step=111000.ckpt
# n_tokens=2685,
#         d_model=24,
#         max_len=1000




def generate_text(input_text, model, tokens_to_idx, idx_to_tokens, max_length=20, device="cuda"):
    try:
        input_tokens = input_text.split()
        model_input = torch.tensor([tokens_to_idx[token] for token in input_tokens if token in tokens_to_idx]).to(device)
    except KeyError as e:
        return f"Unknown token in input text: {e}"

    predicted_ids = []
    for _ in range(len(model_input), max_length):
        predictions = model(model_input)
        predicted_id = torch.argmax(predictions[-1, :]).unsqueeze(0).to(device)
        predicted_ids.append(predicted_id.item())

        model_input = torch.cat((model_input, predicted_id))

    predicted_tokens = [idx_to_tokens[idx] for idx in predicted_ids]
    output_text = " ".join(predicted_tokens)
    return output_text


@app.route('/generate', methods=['POST'])
def generate():
    data = request.json

    # Validate input
    if not all(key in data for key in ['text', 'model_path', 'max_length']):
        return jsonify({"error": "Missing required fields: text, model_path, max_length"}), 400

    text = data['text']
    model_path = data['model_path']
    max_length = int(data['max_length'])

    # Check if model path exists
    if not os.path.exists(model_path):
        return jsonify({"error": f"Model path '{model_path}' does not exist."}), 404

    # Load the model
    try:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = load_model(model_path, device=device)
    except Exception as e:
        return jsonify({"error": f"Failed to load model: {e}"}), 500

    # Generate text
    try:
        output_text = generate_text(text, model, tokens_to_idx, idx_to_tokens, max_length, device=device)
        return jsonify({"input": text, "output": output_text}), 200
    except Exception as e:
        return jsonify({"error": f"Error during text generation: {e}"}), 500


if __name__ == '__main__':
    folder_path = "C:\\Workspace-ML\\text_data"
    text = read_text_files(folder_path)
    all_tokens_list = text_cleaning(text)
    all_tensors_list, vocab_size, tokens_to_idx, idx_to_tokens = tokens_to_tensors(all_tokens_list)

    app.run(debug=True)