from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, declarative_base
from os import environ

# For MySQL database:
# DB_NAME = "gold_standard_v2"
# DB_PASSWORD = environ.get("DB_PASSWORD")
# DB_URL = f"mysql+pymysql://root:{DB_PASSWORD}@localhost:3306/{DB_NAME}"

# For Sqlite database:
DB_URL = "sqlite:///./gold_standard.sqlite"

engine = create_engine(DB_URL)
sessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base = declarative_base()

