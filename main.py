from fastapi import FastAPI, Depends, Query, BackgroundTasks, HTTPException
from sqlalchemy.orm import Session
from db import engine, sessionLocal
import models, schemas, crud
from typing import Annotated
from enum import Enum
from sqlalchemy import func
from keyphrase_models import model_dict
from similarity_models import similarity_dict

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
    by_reviewer_sync: str = "by_reviewer_sync"
    by_reviewer_describe: str = "by_reviewer_describe"


## Check error functions for POST endpoints:
def check_extract_keywords(db, model_name, skip, limit, error):
    try:
        error.status = "processing"
        db.commit()
        crud.extract_papers_keywords(db=db, model_name=model_name, skip=skip, limit=limit)
    except Exception as e:
        error.error = repr(e)
        db.commit()
    finally:
        error.status = "clear"
        db.commit()

def check_compute_similarity(db, model_name, similarity_name, skip, limit, error):
    try:
        error.status = "processing"
        db.commit()
        crud.compute_papers_similarity(db=db, model_name=model_name, similarity_name=similarity_name, skip=skip, limit=limit)
    except Exception as e:
        error.error = repr(e)
        db.commit()
    finally:
        error.status = "clear"
        db.commit()


## Endpoints:
# HOME:
@app.get("/")
def validity_check():
    return {"works":"fine"}

# POST/CREATE:
@app.post("/extract_keywords/{model_name}")#, response_model=schemas.Item)
def extract_keywords(background_tasks: BackgroundTasks,
                    model_name: str,
                    skip: Annotated[int, Query(ge=0, le=3411)] = 0,
                    limit: Annotated[int | None, Query(ge=1, description="If None, then calculates keywords for all papers, else calculates for provided range of papers")] = None,
                    db: Session = Depends(get_db)):
    if model_name not in list(model_dict.keys()):
        return {"message":f"Given model_name does not exist, insert any of the following: {list(model_dict.keys())}"}

    error = db.query(models.Status_and_Error).filter((models.Status_and_Error.task == "extract_keywords") & (models.Status_and_Error.model_name == model_name)).first()

    if error is None:
        record = models.Status_and_Error(task="extract_keywords", model_name=model_name, status="clear", error="No Error as of now")
        db.add(record)
        db.commit()
        error = db.query(models.Status_and_Error).filter((models.Status_and_Error.task == "extract_keywords") & (models.Status_and_Error.model_name == model_name)).first()
    else:
        error.error = "No Error as of now"
        db.commit()

    last_paper_pk = db.query(func.max(models.Model_Paper_Keywords.paper_pk)).filter(models.Model_Paper_Keywords.model_name == model_name).first()[0]
    status = db.query(models.Status_and_Error.status).filter((models.Status_and_Error.task == 'extract_keywords') & (models.Status_and_Error.model_name == model_name)).first()[0]

    if status == "clear":
        if last_paper_pk != 3412:
            if (last_paper_pk is None) or ((last_paper_pk == skip) and (limit is not None)):
                background_tasks.add_task(check_extract_keywords, db=db, model_name=model_name, skip=skip, limit=limit, error=error)
                return {"message": f"Request for Keyword Extraction for the given model '{model_name}' is accepted. Processing in the background. Hit 'status_and_error' endpoint to check status and potential errors."}
            return {"message": f"The last processed paper_pk in the database is {last_paper_pk}, please hit the endpoint again with a skip of {last_paper_pk} and a limit of any positive integer, don't keep it None."}
        return {"message": f"Keywords for all the papers have already been extracted for the given model '{model_name}'"}
    return {"message": f"Keywords extraction for the given model '{model_name}' is already in process for the previous request, please hit 'status_and_error' endpoint to check status."}

@app.post("/compute_similarity/{model_name}/{similarity_name}")
def compute_similarity(background_tasks: BackgroundTasks,
                    model_name: str,
                    similarity_name: str,
                    skip: Annotated[int, Query(ge=0, le=476)] = 0,
                    limit: Annotated[int | None, Query(ge=1, description="If None, then computes similarity for all records, else computes for provided range")] = None,
                    db: Session = Depends(get_db)):
    if similarity_name not in list(similarity_dict.keys()):
        return {"message":f"Given similarity_name does not exist, insert any of the following: {list(similarity_dict.keys())}"}
    available_models = db.query(models.Model_Paper_Keywords.model_name).distinct().all()
    for item in available_models:
        if model_name == item[0]:
            break
    else:
        return {"message":f"Given model_name does not have it's keywords computed, insert any of the following: {available_models}"}

    error = db.query(models.Status_and_Error).filter((models.Status_and_Error.task == "compute_similarity") & (models.Status_and_Error.model_name == model_name) & (models.Status_and_Error.similarity_name == similarity_name)).first()
    
    if error is None:
        record = models.Status_and_Error(task="compute_similarity", model_name=model_name, similarity_name=similarity_name, status="clear", error="No Error as of now")
        db.add(record)
        db.commit()
        error = db.query(models.Status_and_Error).filter((models.Status_and_Error.task == "compute_similarity") & (models.Status_and_Error.model_name == model_name) & (models.Status_and_Error.similarity_name == similarity_name)).first()
    else:
        error.error = "No Error as of now"
        db.commit()

    last_record = len(db.query(models.Model_Reviewer_Paper_Similarity.paper_pk).filter((models.Model_Reviewer_Paper_Similarity.model_name == model_name) & (models.Model_Reviewer_Paper_Similarity.similarity_name == similarity_name)).all())
    last_paper_pk = db.query(func.max(models.Model_Reviewer_Paper_Similarity.paper_pk)).filter((models.Model_Reviewer_Paper_Similarity.model_name == model_name) & (models.Model_Reviewer_Paper_Similarity.similarity_name == similarity_name)).first()[0]
    status = db.query(models.Status_and_Error.status).filter((models.Status_and_Error.task == 'compute_similarity') & (models.Status_and_Error.model_name == model_name) & (models.Status_and_Error.similarity_name == similarity_name)).first()[0]
    
    if status == "clear":
        if last_paper_pk != 3412: # Here also the last record has paper_pk of 3412 (and it's the only one no duplicate) so we can use it
            if (last_paper_pk is None) or ((last_record == skip) and (limit is not None)):
                background_tasks.add_task(check_compute_similarity, db=db, model_name=model_name, similarity_name=similarity_name, skip=skip, limit=limit, error=error)
                return {"message": f"Request for Similarity Computation for the given model '{model_name}' and similarity '{similarity_name}' is accepted. Processing in the background. Hit 'status_and_error' endpoint to check status and potential errors"}
            return {"message": f"The last processed record in the database is {last_record}, please hit the endpoint again with a skip of {last_record} and a limit of any positive integer, don't keep it None."}
        return {"message": f"Similarity for all the records have already been computed for the given model '{model_name}' and similarity '{similarity_name}'"}
    return {"message": f"Similarity computation for the given model '{model_name}' and similarity '{similarity_name}' is already in process for the previous request, please hit 'status_and_error' endpoint to check status."}

# GET/READ:
@app.get("/reviewers/")#, response_model=list[schemas.User])
def get_reviewers(skip: Annotated[int, Query(ge=0, le=57)] = 0,
                limit: Annotated[int | None, Query(ge=1, description="If None, then returns data for all reviewers, else returns data for the provided range")] = None,
                db: Session = Depends(get_db)):
    users = crud.get_reviewers_by_id(db, skip=skip, limit=limit)
    return users

@app.get("/papers/")#, response_model=list[schemas.Item])
def get_papers(skip: Annotated[int, Query(ge=0, le=3411)] = 0,
            limit: Annotated[int | None, Query(ge=1, description="If None, then returns data for all papers, else returns data for the provided range")] = None,
            db: Session = Depends(get_db)):
    items = crud.get_papers_by_id(db, skip=skip, limit=limit)
    return items

@app.get("/keywords/{model_name}")
def get_extracted_keywords(model_name: str,
                        skip: Annotated[int, Query(ge=0, le=3411)] = 0,
                        limit: Annotated[int | None, Query(ge=1, description="If None, then returns data for all records, else returns data for the provided range")] = None,
                        db: Session = Depends(get_db)):
    return crud.get_model_extracted_keywords(db, model_name=model_name, skip=skip, limit=limit)

@app.get("/similarity/{model_name}/{similarity_name}")
def get_similarity_values(model_name: str,
                        similarity_name: str,
                        reviewer_pk: Annotated[int | None, Query(ge=1, le=58, description="[1,58] to get specific data; else get all data by default")] = None,
                        norm: Annotated[bool, Query(description="if True, does min-max normalization else no normalization")] = False,
                        db: Session = Depends(get_db)):
    return crud.get_model_similarity_values(db, model_name=model_name, similarity_name=similarity_name, reviewer_pk=reviewer_pk, norm=norm)

@app.get("/correlation/{model_name}/{similarity_name}")
def get_correlation_values(model_name: str,
                        similarity_name: str,
                        layout: Annotated[LayoutEnum, Query(description="'whole': to get correlation of all reviewers; 'by_reviewer': to get individual correlations")] = LayoutEnum.whole,
                        norm: Annotated[bool, Query(description="if True, does min-max normalization else no normalization")] = False,
                        db: Session = Depends(get_db)):
    return crud.get_model_correlation_values(db, model_name=model_name, similarity_name=similarity_name, layout=layout, norm=norm)

@app.get("/status_and_error/")
def get_status_and_error_messages(db: Session = Depends(get_db)):
    return db.query(models.Status_and_Error) \
        .order_by(models.Status_and_Error.model_name).all()
