from sqlalchemy.orm import Session
from sqlalchemy import desc, distinct
import models, schemas
from keyphrase_models import model_dict
from similarity_models import similarity_dict
from correlations import get_correlations
import numpy as np
# from db import engine, sessionLocal
import matplotlib.pyplot as plt

def get_reviewers_by_id(db: Session, skip: int, limit: int | None):
    if limit is None:
        return db.query(models.Reviewers).all()
    else:
        return db.query(models.Reviewers).offset(skip).limit(limit).all()

def get_papers_by_id(db: Session, skip: int, limit: int | None):
    if limit is None:
        return db.query(models.Papers).all()
    else:
        return db.query(models.Papers).offset(skip).limit(limit).all()

def extract_papers_keywords(db: Session, model_name: str, skip: int, limit: int | None):
    if limit is None: # extracting keywords from every paper
        results = db.query(models.Papers).all()
    else:
        results = db.query(models.Papers).offset(skip).limit(limit).all()

    for result in results:
        title = result.title
        abstract = result.abstract
        pdf_text_path = result.pdf_text_path

        if isinstance(title, str):
            title_keywords = model_dict[model_name](title, 'tit')
        else:
            title_keywords = []

        if isinstance(abstract, str):
            abstract_keywords = model_dict[model_name](abstract, 'abs')
        else:
            abstract_keywords = []

        if isinstance(pdf_text_path, str):
            with open(pdf_text_path, "r", encoding="utf-8") as tf:
                pdf_text = tf.read()
            pdf_text_keywords = model_dict[model_name](pdf_text)
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

def compute_papers_similarity(db: Session, model_name: str, similarity_name: str, skip: int, limit: int | None):
    if limit is None: # Computing similarity of every record
        results = db.query(models.Rating.reviewer_pk, models.Rating.paper_pk).all()
    else:
        results = db.query(models.Rating.reviewer_pk, models.Rating.paper_pk).offset(skip).limit(limit).all()

    reviewer_cache = results[0].reviewer_pk # setting first reviewer
    past_papers_cache = db.query(models.Model_Paper_Keywords).join(models.Reviewers_Papers, models.Reviewers_Papers.paper_pk == models.Model_Paper_Keywords.paper_pk).filter((models.Reviewers_Papers.reviewer_pk == reviewer_cache) & (models.Model_Paper_Keywords.model_name == model_name)).all()
    for result in results:
        reviewed_paper_data = db.query(models.Model_Paper_Keywords).filter(models.Model_Paper_Keywords.paper_pk == result.paper_pk).first()
        # Note: model_paper_keywords.model_keywords_w_pdf can be empty string if none of title, abstract or pdf_text are available
        reviewer = result.reviewer_pk
        if reviewer == reviewer_cache:
            past_papers_data = past_papers_cache
        else:
            past_papers_data = db.query(models.Model_Paper_Keywords).join(models.Reviewers_Papers, models.Reviewers_Papers.paper_pk == models.Model_Paper_Keywords.paper_pk).filter((models.Reviewers_Papers.reviewer_pk == reviewer) & (models.Model_Paper_Keywords.model_name == model_name)).all()
            past_papers_cache = past_papers_data
            reviewer_cache = reviewer

        if similarity_dict[similarity_name]["mode"] == "mean":
            if reviewed_paper_data.model_keywords_wo_pdf:
                similarity_wo_pdf, terms_wo_pdf = 0, 0
                for past_paper in past_papers_data:
                    if past_paper.model_keywords_wo_pdf:
                        similarity_wo_pdf += similarity_dict[similarity_name]["sim"](past_paper.model_keywords_wo_pdf, reviewed_paper_data.model_keywords_wo_pdf)
                        terms_wo_pdf += 1
                if similarity_wo_pdf == 0:
                    average_similarity_wo_pdf = None # can't make it zero, cause then it can be infered as the model giving a similarity of 0 instead of data inavailability
                else:
                    average_similarity_wo_pdf = round(similarity_wo_pdf/terms_wo_pdf, 2)
            else:
                average_similarity_wo_pdf = None # can't make it zero, cause then it can be infered as the model giving a similarity of 0 instead of data inavailability

            if reviewed_paper_data.model_keywords_w_pdf:
                similarity_w_pdf, terms_w_pdf = 0, 0
                for past_paper in past_papers_data:
                    if past_paper.model_keywords_w_pdf:
                        similarity_w_pdf += similarity_dict[similarity_name]["sim"](past_paper.model_keywords_w_pdf, reviewed_paper_data.model_keywords_w_pdf)
                        terms_w_pdf += 1
                if similarity_w_pdf == 0:
                    average_similarity_w_pdf = None # can't make it zero, cause then it can be infered as the model giving a similarity of 0 instead of data inavailability
                else:
                    average_similarity_w_pdf = round(similarity_w_pdf/terms_w_pdf, 2)
            else:
                average_similarity_w_pdf = None # can't make it zero, cause then it can be infered as the model giving a similarity of 0 instead of data inavailability
        
        elif similarity_dict[similarity_name]["mode"] == "max":
            if reviewed_paper_data.model_keywords_wo_pdf:
                max_similarity_wo_pdf = -1
                for past_paper in past_papers_data:
                    if past_paper.model_keywords_wo_pdf:
                        sim = similarity_dict[similarity_name]["sim"](past_paper.model_keywords_wo_pdf, reviewed_paper_data.model_keywords_wo_pdf)
                        if sim > max_similarity_wo_pdf:
                            max_similarity_wo_pdf = sim
                if max_similarity_wo_pdf == -1:
                    average_similarity_wo_pdf = None # can't make it zero, cause then it can be infered as the model giving a similarity of 0 instead of data inavailability
                else:
                    average_similarity_wo_pdf = max_similarity_wo_pdf
            else:
                average_similarity_wo_pdf = None # can't make it zero, cause then it can be infered as the model giving a similarity of 0 instead of data inavailability

            if reviewed_paper_data.model_keywords_w_pdf:
                max_similarity_w_pdf = -1
                for past_paper in past_papers_data:
                    if past_paper.model_keywords_w_pdf:
                        sim = similarity_dict[similarity_name]["sim"](past_paper.model_keywords_w_pdf, reviewed_paper_data.model_keywords_w_pdf)
                        if sim > max_similarity_w_pdf:
                            max_similarity_w_pdf = sim
                if max_similarity_w_pdf == -1:
                    average_similarity_w_pdf = None # can't make it zero, cause then it can be infered as the model giving a similarity of 0 instead of data inavailability
                else:
                    average_similarity_w_pdf = max_similarity_w_pdf
            else:
                average_similarity_w_pdf = None # can't make it zero, cause then it can be infered as the model giving a similarity of 0 instead of data inavailability

        similarity = models.Model_Reviewer_Paper_Similarity(reviewer_pk=result.reviewer_pk, paper_pk=result.paper_pk, model_name=model_name, similarity_name=similarity_name, model_similarity_wo_pdf=average_similarity_wo_pdf, model_similarity_w_pdf=average_similarity_w_pdf)
        db.add(similarity)
        db.commit()

def get_model_extracted_keywords(db: Session, model_name: str, skip: int, limit: int | None):
    available_options = db.query(models.Model_Paper_Keywords.model_name).distinct().all()
    for item in available_options:
        if model_name == item[0]:
            break
    else:
        return {"message":f"available options are {available_options}, choose any from them."}

    if limit is None:
        return db.query(models.Model_Paper_Keywords).filter(models.Model_Paper_Keywords.model_name == model_name).all()
    else:
        return db.query(models.Model_Paper_Keywords).offset(skip).limit(limit).filter(models.Model_Paper_Keywords.model_name == model_name).all()

def get_model_similarity_values(db: Session, model_name: str, similarity_name: str, reviewer_pk: int | None, norm: bool):
    available_options = db.query(models.Model_Reviewer_Paper_Similarity.model_name, models.Model_Reviewer_Paper_Similarity.similarity_name).distinct().all()
    for item in available_options:
        if (model_name == item[0]) and (similarity_name == item[1]):
            break
    else:
        return {"message":f"available options are {available_options}, choose any from them."}

    if reviewer_pk is None:
        results = db.query(models.Rating.reviewer_pk, models.Rating.paper_pk, models.Rating.rating, models.Model_Reviewer_Paper_Similarity.model_similarity_wo_pdf, models.Model_Reviewer_Paper_Similarity.model_similarity_w_pdf) \
            .join(models.Model_Reviewer_Paper_Similarity, (models.Rating.reviewer_pk == models.Model_Reviewer_Paper_Similarity.reviewer_pk) & (models.Rating.paper_pk == models.Model_Reviewer_Paper_Similarity.paper_pk)) \
            .filter((models.Model_Reviewer_Paper_Similarity.model_name == model_name) & (models.Model_Reviewer_Paper_Similarity.similarity_name == similarity_name)) \
            .order_by(desc(models.Model_Reviewer_Paper_Similarity.model_similarity_wo_pdf)).all()
    else:
        results = db.query(models.Rating.reviewer_pk, models.Rating.paper_pk, models.Rating.rating, models.Model_Reviewer_Paper_Similarity.model_similarity_wo_pdf, models.Model_Reviewer_Paper_Similarity.model_similarity_w_pdf) \
            .join(models.Model_Reviewer_Paper_Similarity, (models.Rating.reviewer_pk == models.Model_Reviewer_Paper_Similarity.reviewer_pk) & (models.Rating.paper_pk == models.Model_Reviewer_Paper_Similarity.paper_pk)) \
            .filter((models.Rating.reviewer_pk == reviewer_pk) & (models.Model_Reviewer_Paper_Similarity.model_name == model_name) & (models.Model_Reviewer_Paper_Similarity.similarity_name == similarity_name)) \
            .order_by(desc(models.Model_Reviewer_Paper_Similarity.model_similarity_wo_pdf)).all()

    if norm:
        model_similarity_wo_pdf = []
        model_similarity_w_pdf = []

        model_old_similarity_wo_pdf = [result.model_similarity_wo_pdf for result in results]
        model_old_similarity_w_pdf = [result.model_similarity_w_pdf for result in results]

        minValue_wo_pdf, maxValue_wo_pdf = min(model_old_similarity_wo_pdf), max(model_old_similarity_wo_pdf)
        minValue_w_pdf, maxValue_w_pdf = min(model_old_similarity_w_pdf), max(model_old_similarity_w_pdf)

        for result in results:
            model_similarity_wo_pdf.append(((result.model_similarity_wo_pdf - minValue_wo_pdf)/(maxValue_wo_pdf - minValue_wo_pdf))*(5-0) + 0) # New range [0,5]
            model_similarity_w_pdf.append(((result.model_similarity_w_pdf - minValue_w_pdf)/(maxValue_w_pdf - minValue_w_pdf))*(5-0) + 0)
    else:
        model_similarity_wo_pdf = [result.model_similarity_wo_pdf for result in results]
        model_similarity_w_pdf = [result.model_similarity_w_pdf for result in results]

    return [{"Reviewer_pk":result.reviewer_pk,
            "Paper_pk":result.paper_pk,
            "Rating":result.rating,
            "Model_Similarity_wo_pdf":round(model_similarity_wo_pdf[idx], 2),
            "Model_Similarity_w_pdf":round(model_similarity_w_pdf[idx], 2)}
            for idx, result in enumerate(results)]

def get_model_correlation_values(db: Session, model_name: str, similarity_name: str, layout: str, norm: bool):
    available_options = db.query(models.Model_Reviewer_Paper_Similarity.model_name, models.Model_Reviewer_Paper_Similarity.similarity_name).distinct().all()
    for item in available_options:
        if (model_name == item[0]) and (similarity_name == item[1]):
            break
    else:
        return {"message":f"available options are {available_options}, choose any from them."}

    if layout == "whole":
        output = get_model_similarity_values(db=db, model_name=model_name, similarity_name=similarity_name, reviewer_pk=None, norm=norm)
        rating = [row["Rating"] for row in output]
        model_similarity_wo_pdf = [row["Model_Similarity_wo_pdf"] for row in output]
        model_similarity_w_pdf = [row["Model_Similarity_w_pdf"] for row in output]
        return {"Model_Correlation_wo_pdf": get_correlations(rating, model_similarity_wo_pdf),
                "Model_Correlation_w_pdf": get_correlations(rating, model_similarity_w_pdf)}

    elif layout == "by_reviewer":
        return_result = {}
        for idx in range(1,59,1):
            output = get_model_similarity_values(db=db, model_name=model_name, similarity_name=similarity_name, reviewer_pk=idx, norm=norm)
            rating = [row["Rating"] for row in output]
            model_similarity_wo_pdf = [row["Model_Similarity_wo_pdf"] for row in output]
            model_similarity_w_pdf = [row["Model_Similarity_w_pdf"] for row in output]
            combined_result = {"Model_Correlation_wo_pdf": get_correlations(rating, model_similarity_wo_pdf),
                            "Model_Correlation_w_pdf": get_correlations(rating, model_similarity_w_pdf)}
            return_result[idx] = combined_result
        print(return_result)
        return return_result

    elif layout == "by_reviewer_describe":
        wo_pdf_dict = {"Pearson":[], "Spearman":[], "Kendalltau":[]}
        w_pdf_dict = {"Pearson":[], "Spearman":[], "Kendalltau":[]}
        for idx in range(1,59,1):
            output = get_model_similarity_values(db=db, model_name=model_name, similarity_name=similarity_name, reviewer_pk=idx, norm=norm)
            rating = [row["Rating"] for row in output]
            model_similarity_wo_pdf = [row["Model_Similarity_wo_pdf"] for row in output]
            model_similarity_w_pdf = [row["Model_Similarity_w_pdf"] for row in output]

            wo_pdf_data = get_correlations(rating, model_similarity_wo_pdf)
            wo_pdf_dict["Pearson"].append(wo_pdf_data["Pearson"]["Correlation"])
            wo_pdf_dict["Spearman"].append(wo_pdf_data["Spearman"]["Correlation"])
            wo_pdf_dict["Kendalltau"].append(wo_pdf_data["Kendalltau"]["Correlation"])

            w_pdf_data = get_correlations(rating, model_similarity_w_pdf)
            w_pdf_dict["Pearson"].append(w_pdf_data["Pearson"]["Correlation"])
            w_pdf_dict["Spearman"].append(w_pdf_data["Spearman"]["Correlation"])
            w_pdf_dict["Kendalltau"].append(w_pdf_data["Kendalltau"]["Correlation"])

        plt.scatter(list(range(1,59,1)), wo_pdf_dict["Pearson"])
        title = f"{similarity_name} similarity"
        print(title)
        plt.title(title)
        plt.xlabel("Reviewer ID")
        plt.ylabel("Pearson Correlation")
        plt.savefig(f".\correlation_graphs\{title}.jpg", dpi=400)

        return {"wo_pdf":{"mean":{"Pearson":np.mean(wo_pdf_dict["Pearson"]), "Spearman":np.mean(wo_pdf_dict["Spearman"]), "Kendalltau":np.mean(wo_pdf_dict["Kendalltau"])},
                        "median":{"Pearson":np.median(wo_pdf_dict["Pearson"]), "Spearman":np.median(wo_pdf_dict["Spearman"]), "Kendalltau":np.median(wo_pdf_dict["Kendalltau"])},
                        "std":{"Pearson":np.std(wo_pdf_dict["Pearson"]), "Spearman":np.std(wo_pdf_dict["Spearman"]), "Kendalltau":np.std(wo_pdf_dict["Kendalltau"])}},
                "w_pdf":{"mean":{"Pearson":np.mean(w_pdf_dict["Pearson"]), "Spearman":np.mean(w_pdf_dict["Spearman"]), "Kendalltau":np.mean(w_pdf_dict["Kendalltau"])},
                        "median":{"Pearson":np.median(w_pdf_dict["Pearson"]), "Spearman":np.median(w_pdf_dict["Spearman"]), "Kendalltau":np.median(w_pdf_dict["Kendalltau"])},
                        "std":{"Pearson":np.std(w_pdf_dict["Pearson"]), "Spearman":np.std(w_pdf_dict["Spearman"]), "Kendalltau":np.std(w_pdf_dict["Kendalltau"])}}}

    else:
        raise Exception("Unknown format, enter either 'whole', 'by_reviewer_describe' or 'by_reviewer'")
