# Freebase Setup and Datasets

Before running our code, you must install and configure **Freebase** on your local machine. This guide walks through downloading and setting up Virtuoso, then importing the Freebase knowledge graph.

### Step 1: Download Virtuoso and Freebase RDF data
Download the required tools:
- [Virtuoso Open Source Packages](https://sourceforge.net/projects/virtuoso/files/virtuoso/)
- [Freebase Data (via Google Developers)](https://developers.google.com/freebase?hl=en)

### Step 2: Data Processing

1. First,unzip the Freebase RDF data:
    ```bash
    gunzip -c freebase-rdf-latest.gz > freebase  # Output ~400GB
    ```

2. Filter the English triplets:
    ```bash
    nohup python -u FilterEnglishTriplets.py 0<freebase 1>FilterFreebase 2>log_err &
    # Output ~125GB
    ```
### Step 3: Import Data into Virtuoso

1. Extract the Virtuoso package:
    ```bash
    tar xvpfz virtuoso-opensource.x86_64-generic_glibc25-linux-gnu.tar.gz
    cd virtuoso-opensource/database/
    mv virtuoso.ini.sample virtuoso.ini
    ```

2. Start the Virtuoso server:
    - To start the server in the current terminal:
      ```bash
      ../bin/virtuoso-t -df
      ```
    - To start the server in the background:
      ```bash
      ../bin/virtuoso-t
      ```

3. Access the Virtuoso database:
    ```bash
    ../bin/isql 1111 dba dba
    ```

4. Load the filtered Freebase data:
    ```sql
    ld_dir('.', 'FilterFreebase', 'http://freebase.com');  # Ensure the filtered data is in the database directory
    rdf_loader_run();  # Adjust parameters in virtuoso.ini to speed up the process
    ```

### Step 4: Check Data Loading Status

1. Open a new terminal and check the loading status:
    ```sql
    select * from DB.DBA.load_list;
    ```
    - The `ll_state` field has three possible values:
      - `0`: Dataset not loaded
      - `1`: Dataset loading in progress
      - `2`: Dataset loaded successfully

2. Verify the number of triples:
    ```sparql
    SPARQL SELECT COUNT(*) { ?s ?p ?o };
    ```

# Dataset and Code
The datasets (CWQ, WebQSP, WebQuestions) are provided under the `data/` directory, with their corresponding alias mappings in `cope_alias/`.
Our codes are modified based on the public project [ToG](https://github.com/GasolSun36/ToG) and [PoG](https://github.com/liyichen-cly/PoG/tree/main). We appreciate the authors for making ToG open-sourced. 

# Running
After installing all necessary configurations, you can execute VoG using the following command:
```sh
python main_freebase.py \  
--dataset cwq \ # the dataset
--max_length 4096 \ # the max length of LLMs output
--temperature_exploration 0.3 \ # the temperature in exploration stage
--temperature_reasoning 0.3 \ # the temperature in reasoning stage
--remove_unnecessary_rel True \ # whether removing unnecessary relations
--LLM_type gpt-3.5-turbo \ # the LLM
--opeani_api_keys YOUR_API_KEY \ # your own api keys
```

# Evaluation
We use **Exact Match** as the evaluation metric. After obtaining the final result file, please evaluate the results using the following example command:
```sh
python eval.py \  
--dataset cwq \ # the dataset
--output_file VoG_cwq_gpt-4.jsonl \ # the result file
```
