# Repository Structure and Workflow

## Directory Structure

- `python_scripts/`: Contains scripts for data preparation and also script for utility functions which are reusable across different scripts and projects.
- `shell_scripts/`: Contains scripts for embedding documents and running the main application.
- `output/`: Stores the output files such as embedded documents and processed data.
- `contriever`: The repository includes a modified version of `contriever/`, originally cloned from [Contriever's GitHub repository](https://github.com/facebookresearch/contriever). Search for "Modified" within the `passage_retrieval.py` script under `contriever/` to see the changes.

## Script Descriptions

### Data Preparation

Located under `/python_scripts`, run `python main.py`. It prepares documents into `support_text.jsonl` and creates `(query, answer)` pairs in `data.jsonl`, both stored under `/output`.
- `base_dataset_processor.py`: Contains the base class for getting the dataset and generating the aformentional two `jsonl` file. Users can create their own dataset processors by extending this base class.
- `sciq_processor.py`: An implementation of the base class for the SCI-Q dataset, serving as an example of how to utilize the framework for custom datasets.


### Document Embedding

- `embed_documents.sh`: Located under `/shell_scripts`, this script embeds all documents from `support_text.jsonl`. The embeddings are stored in `/output/contriever_embedding`.

### Document Retrieval

- `contriever_retrieval`: Uses queries to retrieve documents from the embedded files and evaluates the retrieval using the provided answers. (queries, answers) are from `data.json`. 
- The top 100 retrieved documents will be added to each (query, answers) pair into `data.json` under `/output/data_with_supporting_documents`

## Getting Started

To utilize this repository, follow these steps:

### Remove venv folder
If you happen to have `\venv` folder in your repo, please delete it. As later on, the script running will create another one. They may conflict.

### Running the Scripts

Change directory to `shell_scripts` and execute the script:

```bash
cd shell_scripts
./run.sh
```

### Using the final output

The final output will be `data.json` under `/output/data_with_supporting_documents` under 

Because this file is tweeked multiple times when `./run.sh`, it cannot be easily loaded by `json.load`

You may use the function `read_data_json_file` provided in `read_data.py` under `/python_scripts` to load `data.json` as a correct python dictionary

## Side Notes

### Shell Scripts

- `run.sh`: Sets up the environment for the application. Note that the environment setup may not persist globally in your terminal. It's recommended to extract the environment setup commands and configure them separately in your terminal for code testing and development.

For more details on the processes performed by `run.sh`, refer to the script's inline comments.