# /// script
# requires-python = "==3.12"
# dependencies = [
#     "torch",
#     "transformers",
# ]
# ///


from transformers import AutoTokenizer, AutoModel
from transformers import CLIPModel
from pathlib import Path
import argparse


def embeddings(model_dir, text):
    tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True)
    model = AutoModel.from_pretrained(model_dir, local_files_only=True)
    encoded_input = tokenizer(text, return_tensors="pt")
    output = model(**encoded_input)
    return output[0][:, 0, :].flatten().tolist()


def clip_embeddings(model_dir, text):
    tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True)
    model = CLIPModel.from_pretrained(model_dir, local_files_only=True)
    encoded_input = tokenizer(text, return_tensors="pt")
    output = model.text_model(**encoded_input)
    return output.pooler_output.flatten().tolist()


def main(model_dir, text, isclip):
    if isclip:
        values = clip_embeddings(model_dir, text)
    else:
        values = embeddings(model_dir, text)
    print("\n".join([str(x) for x in values]))


# run e.g: `uv run generate.py "./cache/google-bert/bert-base-uncased" "Text to encode"`
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("model_dir", type=str, help="Model local dir")
    parser.add_argument("text", type=str, help="Text to embed")
    parser.add_argument("--isclip", action="store_true", help="Is clip model")
    args = parser.parse_args()
    main(args.model_dir, args.text, args.isclip)
