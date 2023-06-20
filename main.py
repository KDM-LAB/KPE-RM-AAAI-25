from fastapi import FastAPI, Depends, HTTPException
from sqlalchemy.orm import Session
from db import engine, sessionLocal
import models, schemas, crud

# models.Base.metadata.create_all(bind=engine)

app = FastAPI()

# Dependency
def get_db():
    db = sessionLocal()
    try:
        yield db
    finally:
        db.close()

@app.get("/")
def validity_check():
    return {"works":"fine"}

# POST/CREATE:
@app.post("/extract_keywords/{model_name}/")#, response_model=schemas.Item)
def extract_keywords(model_name: str, skip: int = 0, limit: int = 5, db: Session = Depends(get_db)):
    return crud.extract_papers_keywords(db, model_name=model_name, skip=skip, limit=limit)



# GET/READ:
@app.get("/reviewers/")#, response_model=list[schemas.User])
def get_reviewers(skip: int = 0, limit: int = 20, db: Session = Depends(get_db)):
    users = crud.get_reviewers_by_id(db, skip=skip, limit=limit)
    return users

@app.get("/papers/")#, response_model=list[schemas.Item])
def get_papers(skip: int = 0, limit: int = 100, db: Session = Depends(get_db)):
    items = crud.get_papers_by_id(db, skip=skip, limit=limit)
    return items
