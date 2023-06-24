from sqlalchemy.orm import Session
import models, schemas
from keybert_model import keyword_from_paper
from similarity_model import keyword_similarity
from correlations import get_correlations
import numpy as np
# from db import engine, sessionLocal

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

def extract_papers_keywords(db: Session, model_name: str, skip: int = 0, limit: int | None = None):
    if limit is None: # extracting keywords from every paper
        results = db.query(models.Papers).all()
    else:
        results = db.query(models.Papers).offset(skip).limit(limit).all()

    output = {}
    for result in results:
        title = result.title
        abstract = result.abstract
        pdf_text_path = result.pdf_text_path

        if isinstance(title, str):
            title_keywords = keyword_from_paper(title, 'tit')
        else:
            title_keywords = []

        if isinstance(abstract, str):
            abstract_keywords = keyword_from_paper(abstract, 'abs')
        else:
            abstract_keywords = []

        if isinstance(pdf_text_path, str):
            with open(pdf_text_path, "r", encoding="utf-8") as tf:
                pdf_text = tf.read()
            pdf_text_keywords = keyword_from_paper(pdf_text)
        else:
            pdf_text_keywords = []
        
        fused_keywords_wo_pdf = ";".join(abstract_keywords + title_keywords)
        if pdf_text_keywords:
            fused_keywords_w_pdf = ";".join(pdf_text_keywords)
        else:
            fused_keywords_w_pdf = fused_keywords_wo_pdf

        kw = models.Model_Paper_Keywords(paper_pk=result.paper_pk, model_name=model_name, model_keywords_wo_pdf=fused_keywords_wo_pdf, model_keywords_w_pdf=fused_keywords_w_pdf)
        db.add(kw)
        db.commit()
        output[result.paper_pk] = {"fused_keywords_wo_pdf": fused_keywords_wo_pdf, "fused_keywords_w_pdf": fused_keywords_w_pdf}

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

def get_model_similarity_values(db: Session, model_name: str, reviewer_pk: int, norm: bool):
    if reviewer_pk == 0:
        results = db.query(models.Rating.reviewer_pk, models.Rating.paper_pk, models.Rating.rating, models.Model_Reviewer_Paper_Similarity.model_similarity) \
            .join(models.Model_Reviewer_Paper_Similarity, (models.Rating.reviewer_pk == models.Model_Reviewer_Paper_Similarity.reviewer_pk) & (models.Rating.paper_pk == models.Model_Reviewer_Paper_Similarity.paper_pk)) \
            .filter(models.Model_Reviewer_Paper_Similarity.model_name == model_name).all()
    else:
        results = db.query(models.Rating.reviewer_pk, models.Rating.paper_pk, models.Rating.rating, models.Model_Reviewer_Paper_Similarity.model_similarity) \
            .join(models.Model_Reviewer_Paper_Similarity, (models.Rating.reviewer_pk == models.Model_Reviewer_Paper_Similarity.reviewer_pk) & (models.Rating.paper_pk == models.Model_Reviewer_Paper_Similarity.paper_pk)) \
            .filter((models.Rating.reviewer_pk == reviewer_pk) & (models.Model_Reviewer_Paper_Similarity.model_name == model_name)).all()

    if norm:
        model_similarity = []
        model_old_similarity = [result.model_similarity for result in results]
        minValue, maxValue = min(model_old_similarity), max(model_old_similarity)
        for result in results:
            model_similarity.append(((result.model_similarity - minValue)/(maxValue - minValue))*(5-0) + 0)

    else:
        model_similarity = [result.model_similarity for result in results]

    return [{"Reviewer_pk":result.reviewer_pk,
            "Paper_pk":result.paper_pk,
            "Rating":result.rating,
            "Model_similarity":round(model_similarity[idx], 2)}
            for idx, result in enumerate(results)]

def get_model_correlation_values(db: Session, model_name: str, layout: str, norm: bool):
    if layout == "whole":
        output = get_model_similarity_values(db=db, model_name=model_name, reviewer_pk=0, norm=norm)
        rating = [row["Rating"] for row in output]
        model_similarity = [row["Model_similarity"] for row in output]
        return get_correlations(rating, model_similarity)

    elif layout == "by_reviewer":
        return_result = {}
        for idx in range(1,59,1):
            output = get_model_similarity_values(db=db, model_name=model_name, reviewer_pk=idx, norm=norm)
            rating = [row["Rating"] for row in output]
            model_similarity = [row["Model_similarity"] for row in output]
            return_result[idx] = get_correlations(rating, model_similarity)
        return return_result

    else:
        raise Exception("Unknown format, enter either 'whole' or 'by_reviewer'")


