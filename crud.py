from sqlalchemy.orm import Session
import models, schemas
from keybert_model import keyword_from_paper
from similarity_model import keyword_similarity
#from db import engine, sessionLocal

def json_maker(x):
    # this relies on the fact that our sequence contains chunks of similar numbers never repeating
    main_dict = {}
    sub_dict = {}
    initial_val = list(x.keys())[0][0]
    for key, val in x.items():
        if key[0] == initial_val:
            sub_dict[key[1]] = val
        else:
            main_dict[initial_val] = sub_dict.copy()
            sub_dict.clear()
            sub_dict[key[1]] = val
            initial_val = key[0]
    main_dict[initial_val] = sub_dict

    return main_dict

def get_papers_by_id(db: Session, skip: int = 0, limit: int = 100):
    return db.query(models.Papers).offset(skip).limit(limit).all()

def get_reviewers_by_id(db: Session, skip: int = 0, limit: int = 20):
    return db.query(models.Reviewers).offset(skip).limit(limit).all()

def extract_papers_keywords(db: Session, model_name: str, skip: int = 0, limit: int = 5):
    if limit == -1: # extracting keywords from every paper
        results = db.query(models.Papers).all()
    else:
        results = db.query(models.Papers).offset(skip).limit(limit).all()

    output = {}
    for result in results:
        title = result.title
        abstract = result.abstract
        if isinstance(title, str):
            title_keywords = keyword_from_paper(title)
        else:
            title_keywords = []

        if isinstance(abstract, str):
            abstract_keywords = keyword_from_paper(abstract)
        else:
            abstract_keywords = []
        
        fused_keywords = ";".join(abstract_keywords + title_keywords)
        output[result.paper_pk] = fused_keywords
        
        kw = models.Model_Paper_Keywords(paper_pk=result.paper_pk, model_name=model_name, model_keywords=fused_keywords)
        db.add(kw)
        db.commit()

    return output

def compute_papers_similarity(db: Session, model_name: str):
    results = db.query(models.Rating.reviewer_pk, models.Rating.paper_pk).all()
    reviewer_cache = results[0].reviewer_pk # setting first reviewer
    past_papers_cache = db.query(models.Model_Paper_Keywords).join(models.Reviewers_Papers, models.Reviewers_Papers.paper_pk == models.Model_Paper_Keywords.paper_pk).filter((models.Reviewers_Papers.reviewer_pk == reviewer_cache) and (model.Model_Paper_Keywords.model_name == model_name)).all()
    output = {}
    for result in results:
        reviewed_paper_data = db.query(models.Model_Paper_Keywords).filter(models.Model_Paper_Keywords.paper_pk == result.paper_pk).first()
        if reviewed_paper_data.model_keywords:
            reviewer = result.reviewer_pk
            if reviewer == reviewer_cache:
                past_papers_data = past_papers_cache
            else:
                past_papers_data = db.query(models.Model_Paper_Keywords).join(models.Reviewers_Papers, models.Reviewers_Papers.paper_pk == models.Model_Paper_Keywords.paper_pk).filter((models.Reviewers_Papers.reviewer_pk == reviewer) and (model.Model_Paper_Keywords.model_name == model_name)).all()
                past_papers_cache = past_papers_data
                reviewer_cache = reviewer

            similarity = 0
            terms = 0
            for past_paper in past_papers_data:
                if past_paper.model_keywords:
                    similarity += keyword_similarity(past_paper.model_keywords, reviewed_paper_data.model_keywords)
                    terms += 1
            average_similarity = round(similarity/terms, 2)
        else:
            average_similarity = None
        
        output[(result.reviewer_pk, result.paper_pk)] = average_similarity
        similarity = models.Model_Reviewer_Paper_Similarity(reviewer_pk=result.reviewer_pk, paper_pk=result.paper_pk, model_name=model_name, model_similarity=average_similarity)
        db.add(similarity)
        db.commit()

    return json_maker(output)

