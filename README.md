# General Multimodal Embedding for Image Search

This project is developed based on the [GME](https://huggingface.co/Alibaba-NLP/gme-Qwen2-VL-2B-Instruct) model and is used for testing image retrieval under arbitrary inputs.
- Paper: [GME: Improving Universal Multimodal Retrieval by Multimodal LLMs](https://arxiv.org/abs/2412.16855)

![demo](https://github.com/user-attachments/assets/468928dc-1426-4fb4-a6aa-d0f857eabd0e)

## Setup

``` bash
# Set Environment
conda create -n gme python=3.10
conda activate gme
conda install pytorch==2.3.1 torchvision==0.18.1 torchaudio==2.3.1 pytorch-cuda=12.1 -c pytorch -c nvidia
conda install -c pytorch -c nvidia faiss-gpu=1.9.0
pip install transformers                               # test with 4.47.1
pip install gradio                                     # test with 5.9.1
```

``` bash
# Get Model
pip install -U huggingface_hub
export HF_ENDPOINT=https://hf-mirror.com
huggingface-cli download Alibaba-NLP/gme-Qwen2-VL-2B-Instruct --local-dir gme-Qwen2-VL-2B-Instruct
```

## How to Use

1.  **Prepare the database** for retrieval, use [build_index.py](build_index.py) for feature extraction and index building.
2.  **run** [retrieval_app.py](retrieval_app.py) for online retrieval.


<details>
  <summary>Detailed usage</summary>

```bash
usage: build_index.py [-h] [--model_path MODEL_PATH] [--image_dir IMAGE_DIR] [--batch_size BATCH_SIZE] [--embeddings_output EMBEDDINGS_OUTPUT] [--index_output INDEX_OUTPUT] [--image_paths_output IMAGE_PATHS_OUTPUT]
options:
  --model_path MODEL_PATH
                        Path to the GmeQwen2VL model.
  --image_dir IMAGE_DIR
                        Path to the directory containing new images.
  --batch_size BATCH_SIZE
                        Batch size for embedding extraction.
  --embeddings_output EMBEDDINGS_OUTPUT
                        Output file for saving image embeddings.
  --index_output INDEX_OUTPUT
                        Output file for saving FAISS index.
  --image_paths_output IMAGE_PATHS_OUTPUT
                        Output file for saving image paths.

usage: retrieval_app.py [-h] [--model_path MODEL_PATH] [--image_embeddings_file IMAGE_EMBEDDINGS_FILE] [--faiss_index_file FAISS_INDEX_FILE] [--image_paths_file IMAGE_PATHS_FILE]
options:
  --model_path MODEL_PATH
                        Path to the GME model.
  --image_embeddings_file IMAGE_EMBEDDINGS_FILE
                        Path to the image embeddings file.
  --faiss_index_file FAISS_INDEX_FILE
                        Path to the FAISS index file.
  --image_paths_file IMAGE_PATHS_FILE
                        Path to the file containing image paths.        
```
</details>

## Results

- [gallery.zip](https://github.com/BIGBALLON/GME-Search/releases/download/v0.1.0/gallery.zip) : the set of images used to build the database(**1,131** images).
- [query.zip](https://github.com/BIGBALLON/GME-Search/releases/download/v0.1.0/query.zip) : some query images(17 images).
- Below are some test results along with their visualizations.
- Test with GeForce RTX 4070 Ti SUPER(16GB) on WSL2.

<details>
  <summary><strong>Image(+Text) -> Image</strong></summary>

  <video src="https://github.com/user-attachments/assets/b92e9782-5873-4f2d-9fe9-4d6aecd2ccfc"></video>
  
</details>
<details>
  <summary><strong>Text -> Image[Chinese input]</strong></summary>

<video src="https://github.com/user-attachments/assets/c8efe5fb-4d0d-46dc-9a17-b1aa0bd88572"></video>

</details>

<details>
  <summary><strong>Text -> Image[English input]</strong></summary>

<video src="https://github.com/user-attachments/assets/492b6aa2-3ba2-4337-8d5e-1ba0ab5b997e"></video>

</details>
<details>
  <summary><strong>Text(long) -> Image</strong></summary>

<video src="https://github.com/user-attachments/assets/49e57772-8846-4cf4-bc28-004337234228"></video>

</details>

## License

This project is released under the [MIT License](./LICENSE).

## Citation

```
@misc{li2025gmesearch,
  author = {Wei Li},
  title = {Gradio app with GME for Image Search},
  howpublished = {\url{https://github.com/BIGBALLON/GME-Search}},
  year = {2025}
}
```

Please create a pull request if you find any bugs or want to contribute code. :smile:
