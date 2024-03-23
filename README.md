### Getting Started:
First you need to create a new MySQL database named "gold_standard" using MYSQL command line client or any other application. If you want to name it diffrently, then you have to change `DB_NAME = "gold_standard"` line in `db.py` file with you own database name. If you want to use sqLite, then no need to follow the above steps and just change `MYSQL_DB_URL = f"mysql+pymysql://root:{DB_PASSWORD}@localhost:3306/{DB_NAME}"` line in `db.py` file according to sqLite requirements.

Install the dependencies from `requrirements.txt`. You might also want to download "spacy en_core_web_md" pipline by running `python -m spacy download en_core_web_md` in the terminal, if not already installed by `requirements.txt` file. Download the gold standard dataset folder from [gold_standard_dataset](https://github.com/niharshah/goldstandard-reviewer-paper-match) and add it in the project folder. Make sure to name it "data" if not already.

Run the db_populate.py file first to create and populate the database with gold_standard_data. After that run `uvicorn main:app --reload` in terminal to start the backend server (`uvicorn main:app`, if you want to start without debug mode). Now you can visit `localhost:8000/docs` to interact with the backend and database. You can extract keywords, compute similarities, compute correlation and many more.

### Some model related extra steps:
#### 1. For Promptrank:
- You need to install java into your system to extract keyphrases using the promptrank model. You can install it from [here](https://www.oracle.com/java/technologies/downloads/#jdk21-windows). JDK21 or JDK19 should work.
- You also need to download Standord Core NLP from [here](https://stanfordnlp.github.io/CoreNLP/download.html). After extracting the file, you need to move it into the folder `keyphrase_models/promptrank/`.
- Running promptrank model takes a lot of time, hence for gold_standard_dataset, we have already extracted keyphrases and put it inside `promptrank_keyphrases` folder. So if you just have to populate the promptrank keyphrases for gold_standard_dataset, you can use these precomputed keyphrases. All you need to do is comment/uncomment lines in the files `keyphrases_models/__inin__.py` and `crud.py` as mentioned in the files.

#### 2. For bertkpe:
- You need to download the zip file (BertKPE.zip) from [here](https://drive.google.com/drive/folders/1qDUtiR3QtNYVPfpIfjWiO_iFwCw_gCzT?usp=sharing). Extract the file and move the folders into `keyphrase_models/BertKPE`. Ensure that after extracting, the two folders you get are named "checkpoints" and "data".

#### 3. For one2set:
- You need to download the zip file (kg_one2set.zip) from [here](https://drive.google.com/drive/folders/1qDUtiR3QtNYVPfpIfjWiO_iFwCw_gCzT?usp=sharing). Extract the file and move the folders into `keyphrase_models/kg_one2set`. Ensure that after extracting, the two folders you get are named "output" and "data".

### Some details regarding the models:
Every model is made to extract 30 keyphrases, provided there is an argument controlling the number of extracted keyphrases, else it's upon the model. Every model extracts multiword keyphrases, unless provided otherwise. Every model which have been used to extract keyphrases are:

| Name   |      Description      |
|:----------:|:-------------:|
| TF-IDF | Statistical |
| textrank | Graphical |
| kpminer | Statistical |
| singlerank | Graphical |
| positionrank | Graphical |
| yake | Statistical |
| multipartiterank | Graphical |
| topicrank | Graphical |
| keybert | DL. 15 keywords. Only single word keyphrases |
| keybert_multiword | DL. 15 keywords, non english papers removed with langdetect |
| keybert_30 | DL. 30 keywords |
| patternrank | DL. 30 keywords, non english papers removed with langdetect |
| patternrank_wo_langdetect | DL. 30 keywords |
| keybart | DL |
| bertkpe | DL |
| one2set | DL |
| promptrank | DL |

Python version `3.10.11` is used in the project.
