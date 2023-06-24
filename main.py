from fastapi import FastAPI, Depends, Query
from sqlalchemy.orm import Session
from db import engine, sessionLocal
import models, schemas, crud
from typing import Annotated
from enum import Enum

# models.Base.metadata.create_all(bind=engine)

app = FastAPI()

# Dependency
def get_db():
    db = sessionLocal()
    try:
        yield db
    finally:
        db.close()

class LayoutEnum(str, Enum):
    whole: str = "whole"
    by_reviewer: str = "by_reviewer"

@app.get("/")
def validity_check():
    return {"works":"fine"}

# POST/CREATE:
@app.post("/extract_keywords/{model_name}/")#, response_model=schemas.Item)
def extract_keywords(model_name: str,
                    skip: Annotated[int, Query(ge=0, le=3411)] = 0,
                    limit: Annotated[int | None, Query(ge=0, description="If None, then calculates keywords for all papers, else calculates for provided range of papers")] = None,
                    db: Session = Depends(get_db)):
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
def get_similarity_values(model_name: str,
                        reviewer_pk: Annotated[int | None, Query(ge=1, le=58, description="[1,58] to get specific data; else get all data by default")] = None,
                        norm: Annotated[bool, Query(description="if True, does min-max normalization else no normalization")] = False,
                        db: Session = Depends(get_db)):
    return crud.get_model_similarity_values(db, model_name=model_name, reviewer_pk=reviewer_pk, norm=norm)

@app.get("/correlation/{model_name}")
def get_correlation_values(model_name: str,
                        layout: Annotated[LayoutEnum, Query(description="'whole': to get correlation of all reviewers; 'by_reviewer': to get individual correlations")] = LayoutEnum.whole,
                        norm: Annotated[bool, Query(description="if True, does min-max normalization else no normalization")] = False,
                        db: Session = Depends(get_db)):
    return crud.get_model_correlation_values(db, model_name=model_name, layout=layout, norm=norm)
