# /// script
# requires-python = "==3.12"
# dependencies = [
#     "torch",
#     "transformers",
#     "model2vec",
# ]
# ///


from transformers import AutoTokenizer, AutoModel
from transformers import CLIPModel
from model2vec import StaticModel
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


def model2vec_embeddings(model_dir, text):
    model = StaticModel.from_pretrained(model_dir)
    output = model.encode(text)
    return output.flatten().tolist()


def main(model_dir, text, emb_type="bert"):
    if emb_type == "bert" or emb_type == "xlm-roberta":
        values = embeddings(model_dir, text)
    elif emb_type == "clip":
        values = clip_embeddings(model_dir, text)
    elif emb_type == "model2vec":
        values = model2vec_embeddings(model_dir, text)
    else:
        raise ValueError(f"Unknown emb_type: {emb_type}")
    print("\n".join([str(x) for x in values]))


# run e.g: `uv run generate.py "./cache/google-bert/bert-base-uncased" "Text to encode"` bert
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("model_dir", type=str, help="Model local dir")
    parser.add_argument("text", type=str, help="Text to embed")
    parser.add_argument("type", type=str, help="Embedding type")
    args = parser.parse_args()
    main(args.model_dir, args.text, args.type)
