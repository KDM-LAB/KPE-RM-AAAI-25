Install the dependencies from `requrirements.txt`. You might also want to download "spacy en_core_web_md" pipline by writing `python -m spacy download en_core_web_md` in the terminal. Python version `3.10.11` is used.

Run the db_populate.py file first to create and populate the database with gold_standard_data. After that run `uvicorn main:app --reload` in terminal to start the backend server.