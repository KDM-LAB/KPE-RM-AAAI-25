First you need to create a new MySQL database named "gold_standard" using MYSQL command line client or any other application. If you want to name it diffrently, then you have to change `DB_NAME = "gold_standard"` line in `db.py` file with you own database name. If you want to use sqLite, then no need to follow the above steps and just change `MYSQL_DB_URL = f"mysql+pymysql://root:{DB_PASSWORD}@localhost:3306/{DB_NAME}"` line in `db.py` file according to sqLite requirements.

Install the dependencies from `requrirements.txt`. You might also want to download "spacy en_core_web_md" pipline by writing `python -m spacy download en_core_web_md` in the terminal, it not already installed by `requirements.txt` file. Download the gold standard dataset folder from [gold_standard_dataset](https://github.com/niharshah/goldstandard-reviewer-paper-match) and add it in the project folder.

Run the db_populate.py file first to create and populate the database with gold_standard_data. After that run `uvicorn main:app --reload` in terminal to start the backend server.

Python version `3.10.11` is used in the project.