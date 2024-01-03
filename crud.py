from sqlalchemy.orm import Session
from sqlalchemy import desc, distinct
import models, schemas
from keyphrase_models import model_dict
from similarity_models import similarity_dict
from correlations import get_correlations, get_correlations_sync
import numpy as np
# from db import engine, sessionLocal
import matplotlib.pyplot as plt
import nltk
from nltk.corpus import stopwords
nltk.download('stopwords')
stop = stopwords.words('english')
import time
import json
from langdetect import detect
import kendalltau_loss

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

        if model_name == "bertkpe":
            Title = title if isinstance(title, str) else ""
            Abstract = abstract if isinstance(abstract, str) else ""

            if Title or Abstract:
                try:
                    fused_keywords_wo_pdf = model_dict[model_name](title=Title, abstract=Abstract)
                except:
                    print(f"Issue In extraction. title: {Title} || abstract: {Abstract}")
                    fused_keywords_wo_pdf = ";".join([i.lower() for i in Title.split(" ") if i.lower() not in stop])
            else:
                fused_keywords_wo_pdf = ""

            if isinstance(pdf_text_path, str):
                with open(pdf_text_path, "r", encoding="utf-8") as tf:
                    pdf_text = tf.read()
                fused_keywords_w_pdf = model_dict[model_name](pdf_text=pdf_text)
            else:
                fused_keywords_w_pdf = fused_keywords_wo_pdf

        elif model_name == "one2set":
            print(f"paper_pk: {result.paper_pk}", end=" | ")
            Title = title if isinstance(title, str) else ""
            Abstract = abstract if isinstance(abstract, str) else ""

            sta = time.perf_counter()
            if Title or Abstract:
                if Title and Abstract:
                    text = Title + " <eos> " + Abstract
                    fused_keywords_wo_pdf = model_dict[model_name](text=text)
                else:
                    text = Title + Abstract
                    fused_keywords_wo_pdf = model_dict[model_name](text=text)
            else:
                fused_keywords_wo_pdf = ""
            eta = time.perf_counter()
            print(f"titabs: {round(eta-sta, 4)}", end=" | ")

            sp = time.perf_counter()
            if isinstance(pdf_text_path, str):
                with open(pdf_text_path, "r", encoding="utf-8") as tf:
                    pdf_text = tf.read()
                fused_keywords_w_pdf = model_dict[model_name](text=pdf_text)
            else:
                fused_keywords_w_pdf = fused_keywords_wo_pdf
            ep = time.perf_counter()
            print(f"pdf: {round(ep-sp, 4)}", end=" | ")

        # Uncomment this elif block if you wish to populate promptrank keyphrases using precomputed keyphrases
        # elif model_name == "promptrank":
        #     ssid = result.ssId
        #     with open(r'promptrank_keyphrases\promptrank_keywords.json', encoding="utf-8", mode="r") as f:
        #         promptrank_dict = json.load(f)

        #     try:
        #         fused_keywords_wo_pdf = promptrank_dict[ssid]["ta_keywords"]
        #     except:
        #         fused_keywords_wo_pdf = ""

        #     try:
        #         fused_keywords_w_pdf = promptrank_dict[ssid]["pdf_keywords"]
        #         if fused_keywords_w_pdf == "":
        #             fused_keywords_w_pdf = fused_keywords_wo_pdf
        #     except:
        #         fused_keywords_w_pdf = fused_keywords_wo_pdf

        else:
            print(f"paper_pk: {result.paper_pk}", end=" | ")
            if isinstance(title, str):
                # if detect(title) == "en":
                st = time.perf_counter()
                try:
                    title_keywords = model_dict[model_name](title, 'tit')
                except:
                    title_keywords = []
                et = time.perf_counter()
                print(f"title: {round(et-st, 4)}", end=" | ")
                # else:
                #     title_keywords = []
                #     print(f"TIT {result.paper_pk} is REJECTED: {title}")
            else:
                title_keywords = []

            if isinstance(abstract, str):
                # if detect(abstract) == "en":
                sa = time.perf_counter()
                abstract_keywords = model_dict[model_name](abstract, 'abs')
                ea = time.perf_counter()
                print(f"abstract: {round(ea-sa, 4)}", end=" | ")
                # else:
                #     abstract_keywords = []
                #     print(f"ABS {result.paper_pk} is REJECTED: {abstract}")
            else:
                abstract_keywords = []

            if isinstance(pdf_text_path, str):
                sr = time.perf_counter()
                with open(pdf_text_path, "r", encoding="utf-8") as tf:
                    pdf_text = tf.read()
                er = time.perf_counter()
                print(f"pdf_read: {round(er-sr, 4)}", end=" | ")
                
                # if detect(pdf_text[:1000]) == "en":
                sp = time.perf_counter()
                pdf_text_keywords = model_dict[model_name](pdf_text)
                ep = time.perf_counter()
                print(f"pdf: {round(ep-sp, 4)}", end=" | ")
                # else:
                #     pdf_text_keywords = []
                #     print(f"PDF {result.paper_pk} is REJECTED: {pdf_text[:100]}")
            else:
                pdf_text_keywords = []
            
            fused_keywords_wo_pdf = ";".join(abstract_keywords + title_keywords)
            if pdf_text_keywords:
                fused_keywords_w_pdf = ";".join(pdf_text_keywords)
            else:
                fused_keywords_w_pdf = fused_keywords_wo_pdf

        sd = time.perf_counter()
        kw = models.Model_Paper_Keywords(paper_pk=result.paper_pk, model_name=model_name, model_keywords_wo_pdf=fused_keywords_wo_pdf, model_keywords_w_pdf=fused_keywords_w_pdf)
        db.add(kw)
        db.commit()
        ed = time.perf_counter()
        print(f"data_write: {round(ed-sd, 4)}")


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
                if terms_wo_pdf == 0:
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
                if terms_w_pdf == 0:
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
            model_similarity_wo_pdf.append(((result.model_similarity_wo_pdf - minValue_wo_pdf)/(maxValue_wo_pdf - minValue_wo_pdf + 1e-2))*(5-0) + 0) # New range [0,5]
            model_similarity_w_pdf.append(((result.model_similarity_w_pdf - minValue_w_pdf)/(maxValue_w_pdf - minValue_w_pdf + 1e-2))*(5-0) + 0)
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
        return return_result

    elif layout == "by_reviewer_sync":
        wo_pdf_corr = {"pearson":[], "spearman":[], "kendaltau":[]}
        w_pdf_corr = {"pearson":[], "spearman":[], "kendaltau":[]}
        for idx in range(1,59,1):
            output = get_model_similarity_values(db=db, model_name=model_name, similarity_name=similarity_name, reviewer_pk=idx, norm=norm)
            rating = [row["Rating"] for row in output]
            model_similarity_wo_pdf = [row["Model_Similarity_wo_pdf"] for row in output]
            model_similarity_w_pdf = [row["Model_Similarity_w_pdf"] for row in output]

            p, s, k = get_correlations_sync(rating, model_similarity_wo_pdf)
            wo_pdf_corr["pearson"].append(p)
            wo_pdf_corr["spearman"].append(s)
            wo_pdf_corr["kendaltau"].append(k)

            p, s, k = get_correlations_sync(rating, model_similarity_w_pdf)
            w_pdf_corr["pearson"].append(p)
            w_pdf_corr["spearman"].append(s)
            w_pdf_corr["kendaltau"].append(k)

        for corr in [wo_pdf_corr, w_pdf_corr]:
            for k in corr.keys():
                corr[k] = np.mean(corr[k])

        return {"Model_Correlation_wo_pdf": wo_pdf_corr,
                "Model_Correlation_w_pdf": w_pdf_corr}

    elif layout == "by_reviewer_describe":
        wo_pdf_dict = {"Pearson":[], "Spearman":[], "Kendalltau":[]}
        w_pdf_dict = {"Pearson":[], "Spearman":[], "Kendalltau":[]}

        refs = {} # for kendalltau loss
        preds_wo_pdf = {} # for kendalltau loss
        preds_w_pdf = {} # for kendalltau loss

        for idx in range(1,59,1):
            output = get_model_similarity_values(db=db, model_name=model_name, similarity_name=similarity_name, reviewer_pk=idx, norm=norm)

            rating = [row["Rating"] for row in output]
            model_similarity_wo_pdf = [row["Model_Similarity_wo_pdf"] for row in output]
            model_similarity_w_pdf = [row["Model_Similarity_w_pdf"] for row in output]
            paper_id = [str(row["Paper_pk"]) for row in output]

            refs[idx] = dict(zip(paper_id, rating))
            preds_wo_pdf[idx] = dict(zip(paper_id, model_similarity_wo_pdf))
            preds_w_pdf[idx] = dict(zip(paper_id, model_similarity_w_pdf))

            wo_pdf_data = get_correlations(rating, model_similarity_wo_pdf)
            wo_pdf_dict["Pearson"].append(wo_pdf_data["Pearson"]["Correlation"])
            wo_pdf_dict["Spearman"].append(wo_pdf_data["Spearman"]["Correlation"])
            wo_pdf_dict["Kendalltau"].append(wo_pdf_data["Kendalltau"]["Correlation"])

            w_pdf_data = get_correlations(rating, model_similarity_w_pdf)
            w_pdf_dict["Pearson"].append(w_pdf_data["Pearson"]["Correlation"])
            w_pdf_dict["Spearman"].append(w_pdf_data["Spearman"]["Correlation"])
            w_pdf_dict["Kendalltau"].append(w_pdf_data["Kendalltau"]["Correlation"])

        # Uncomment to plot and save a graph for pearson correlation of different reviewers
        # plt.figure()
        # plt.scatter(list(range(1,59,1)), wo_pdf_dict["Pearson"])
        # title = f"{model_name} model with {similarity_name} similarity"
        # plt.title(title)
        # plt.xlabel("Reviewer ID")
        # plt.ylabel("Pearson Correlation")
        # plt.savefig(f".\correlation_graphs\{title}.jpg", dpi=400)

        ktl_wo_pdf = kendalltau_loss.compute_main_metric(preds_wo_pdf, refs)
        ktl_w_pdf = kendalltau_loss.compute_main_metric(preds_w_pdf, refs)

        return {"wo_pdf":{"kendalTauLoss":ktl_wo_pdf,
                        "mean":{"Pearson":np.mean(wo_pdf_dict["Pearson"]), "Spearman":np.mean(wo_pdf_dict["Spearman"]), "Kendalltau":np.mean(wo_pdf_dict["Kendalltau"])},
                        "median":{"Pearson":np.median(wo_pdf_dict["Pearson"]), "Spearman":np.median(wo_pdf_dict["Spearman"]), "Kendalltau":np.median(wo_pdf_dict["Kendalltau"])},
                        "std":{"Pearson":np.std(wo_pdf_dict["Pearson"]), "Spearman":np.std(wo_pdf_dict["Spearman"]), "Kendalltau":np.std(wo_pdf_dict["Kendalltau"])}},
                "w_pdf":{"kendalTauLoss":ktl_w_pdf,
                        "mean":{"Pearson":np.mean(w_pdf_dict["Pearson"]), "Spearman":np.mean(w_pdf_dict["Spearman"]), "Kendalltau":np.mean(w_pdf_dict["Kendalltau"])},
                        "median":{"Pearson":np.median(w_pdf_dict["Pearson"]), "Spearman":np.median(w_pdf_dict["Spearman"]), "Kendalltau":np.median(w_pdf_dict["Kendalltau"])},
                        "std":{"Pearson":np.std(w_pdf_dict["Pearson"]), "Spearman":np.std(w_pdf_dict["Spearman"]), "Kendalltau":np.std(w_pdf_dict["Kendalltau"])}}}

    else:
        raise Exception("Unknown format, enter either 'whole', 'by_reviewer_sync', 'by_reviewer_describe' or 'by_reviewer'")
