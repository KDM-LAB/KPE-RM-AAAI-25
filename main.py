from fastapi import FastAPI, Depends, Query, BackgroundTasks, HTTPException
from sqlalchemy.orm import Session
from db import engine, sessionLocal
import models, schemas, crud
from typing import Annotated
from enum import Enum
from fastapi.responses import JSONResponse

# models.Base.metadata.create_all(bind=engine) # Not creating db here, creating in db_populate file

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


## Check error functions for POST endpoints:
def check_extract_keywords(db, model_name, skip, limit, error):
    try:
        crud.extract_papers_keywords(db=db, model_name=model_name, skip=skip, limit=limit)
    except Exception as e:
        error.error = repr(e)
        db.commit()

def check_compute_similarity(db, model_name, error):
    try:
        crud.compute_papers_similarity(db=db, model_name=model_name)
    except Exception as e:
        error.error = repr(e)
        db.commit()

## Endpoints:
# HOME:
@app.get("/")
def validity_check():
    return {"works":"fine"}

# POST/CREATE:
@app.post("/extract_keywords/{model_name}/")#, response_model=schemas.Item)
def extract_keywords(background_tasks: BackgroundTasks,
                    model_name: str,
                    skip: Annotated[int, Query(ge=0, le=3411)] = 0,
                    limit: Annotated[int | None, Query(ge=0, description="If None, then calculates keywords for all papers, else calculates for provided range of papers")] = None,
                    db: Session = Depends(get_db)):
    error = db.query(models.Status_and_Error).filter((models.Status_and_Error.task == "extract_keywords") & (models.Status_and_Error.model_name == model_name)).first()
    if error is None:
        record = models.Status_and_Error(task="extract_keywords", model_name=model_name, status="clear", error="No Error")
        db.add(record)
        db.commit()
    else:
        error.error = "No Error as of now"
        db.commit()
    background_tasks.add_task(check_extract_keywords, db=db, model_name=model_name, skip=skip, limit=limit, error=error)
    return {"message": f"Request for Keyword Extraction for the given model '{model_name}' is accepted. Processing in the background. Hit 'status_and_error' endpoint to check status and potential errors"}

@app.post("/compute_similarity/{model_name}/")
def compute_similarity(background_tasks: BackgroundTasks,
                    model_name: str,
                    db: Session = Depends(get_db)):
    error = db.query(models.Status_and_Error).filter((models.Status_and_Error.task == "compute_similarity") & (models.Status_and_Error.model_name == model_name)).first()
    if error is None:
        record = models.Status_and_Error(task="compute_similarity", model_name=model_name, status="clear", error="No Error")
        db.add(record)
        db.commit()
    else:
        error.error = "No Error as of now"
        db.commit()
    background_tasks.add_task(check_compute_similarity, db=db, model_name=model_name, error=error)
    return {"message": f"Request for Similarity Computation for the given model '{model_name}' is accepted. Processing in the background. Hit 'status_and_error' endpoint to check status and potential errors"}

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

@app.get("/status_and_error/")
def get_status_and_error_messages(db: Session = Depends(get_db)):
    return db.query(models.Status_and_Error) \
        .order_by(models.Status_and_Error.model_name).all()
