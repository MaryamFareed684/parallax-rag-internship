def main():
    import torch
    if torch.cuda.is_available():
        print("GPU:", torch.cuda.get_device_name(0))
    else:
        print("WARNING: CUDA/GPU not available, running on CPU")

    import pandas, sentence_transformers, chromadb, spacy
    print("All imports OK")

if __name__ == "__main__":
    main()