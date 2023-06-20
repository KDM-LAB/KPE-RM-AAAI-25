from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, declarative_base
from os import environ

DB_PASSWORD = environ.get("DB_PASSWORD")
MYSQL_DB_URL = f"mysql+pymysql://root:{DB_PASSWORD}@localhost:3306/gold_standard"

engine = create_engine(MYSQL_DB_URL)
sessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base = declarative_base()

