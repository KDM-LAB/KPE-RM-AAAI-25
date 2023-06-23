from fastapi import FastAPI, Depends, Query
from sqlalchemy.orm import Session
from db import engine, sessionLocal
import models, schemas, crud
from typing import Annotated

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

@app.post("/compute_similarity/{model_name}/")
def compute_similarity(model_name: str, db: Session = Depends(get_db)):
    return crud.compute_papers_similarity(db, model_name=model_name)

# GET/READ:
@app.get("/reviewers/")#, response_model=list[schemas.User])
def get_reviewers(skip: int = 0, limit: int = 20, db: Session = Depends(get_db)):
    users = crud.get_reviewers_by_id(db, skip=skip, limit=limit)
    return users

@app.get("/papers/")#, response_model=list[schemas.Item])
def get_papers(skip: int = 0, limit: int = 100, db: Session = Depends(get_db)):
    items = crud.get_papers_by_id(db, skip=skip, limit=limit)
    return items

@app.get("/similarity/{model_name}")
def get_similarity_values(model_name: str, \
                        reviewer_pk: Annotated[int, Query(ge=0, le=58, description="0 to get all reviewer's data; [1,58] to get specific data")] = 0, \
                        norm: Annotated[str | None, Query(description="'min_max' or 'z-score'; no normalization by default")] = None, \
                        db: Session = Depends(get_db)):
    return crud.get_model_similarity_values(db, model_name=model_name, reviewer_pk=reviewer_pk, norm=norm)
