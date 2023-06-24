import os
import json
import pandas as pd
import numpy as np
import models
from db import engine, sessionLocal

models.Base.metadata.create_all(bind=engine)
db = sessionLocal()

REVIEWERS_PARENT_DIR = r".\data\participants"
PAPERS_PARENT_DIR = r".\data\papers"
RATING_PATH = r".\data\evaluations.csv"
PDF_TEXT_DIR = r".\data\txts"

reviewers_json = os.listdir(REVIEWERS_PARENT_DIR)

for reveiwer in reviewers_json:
    reveiwer_path = os.path.join(REVIEWERS_PARENT_DIR, reveiwer)
    with open(reveiwer_path, "r") as f:
        reveiwer_file = json.load(f)
        reviewer = models.Reviewers(author_id=reveiwer_file["authorId"], name=reveiwer_file["name"])
        db.add(reviewer)
        db.commit()
        db.refresh(reviewer)
        for published_paper in reveiwer_file["papers"]:
            paper_path = os.path.join(PAPERS_PARENT_DIR, published_paper["paperId"]+".json")
            with open(paper_path, "r") as pf:
                paper_file = json.load(pf)

                does_already_exists = db.query(models.Papers).filter(models.Papers.ssId == paper_file["ssId"]).first()
                if does_already_exists:
                    reviewer_paper = models.Reviewers_Papers(reviewer_pk=reviewer.reviewer_pk, paper_pk=does_already_exists.paper_pk)
                    db.add(reviewer_paper)
                    db.commit()
                    db.refresh(reviewer_paper)
                else:
                    text_data_path = None
                    paper_pdf_text_path = os.path.join(PDF_TEXT_DIR, published_paper["paperId"]+".txt")
                    if os.path.exists(paper_pdf_text_path):
                        text_data_path = paper_pdf_text_path
                    paper = models.Papers(ssId=paper_file["ssId"], title=paper_file["title"], abstract=paper_file["abstract"], pdf_text_path=text_data_path, year=paper_file["year"], is_submitted=False)
                    db.add(paper)
                    db.commit()
                    db.refresh(paper)
                    reviewer_paper = models.Reviewers_Papers(reviewer_pk=reviewer.reviewer_pk, paper_pk=paper.paper_pk)
                    db.add(reviewer_paper)
                    db.commit()
                    db.refresh(reviewer_paper)

rating_data = pd.read_csv(RATING_PATH, sep='\t')

for r in range(rating_data.shape[0]):
    auth_id = rating_data.iloc[r, 0]
    auth_pk = db.query(models.Reviewers).filter(models.Reviewers.author_id == auth_id).first().reviewer_pk

    for c in range(1, rating_data.shape[1]-10):
        submitted_paper = rating_data.iloc[r, c]
        if isinstance(submitted_paper, str):
            submitted_paper_path = os.path.join(PAPERS_PARENT_DIR, submitted_paper+".json")
            with open(submitted_paper_path, "r") as sf:
                submitted_paper_file = json.load(sf)
                text_data_path = None
                paper_pdf_text_path = os.path.join(PDF_TEXT_DIR, submitted_paper+".txt")
                if os.path.exists(paper_pdf_text_path):
                    text_data_path = paper_pdf_text_path
                paper = models.Papers(ssId=submitted_paper_file["ssId"], title=submitted_paper_file["title"], abstract=submitted_paper_file["abstract"], pdf_text_path=text_data_path, year=submitted_paper_file["year"], is_submitted=True)
                db.add(paper)
                db.commit()
                db.refresh(paper)
                rate = models.Rating(reviewer_pk=auth_pk, paper_pk=paper.paper_pk, rating=rating_data.iloc[r, c+10])
                db.add(rate)
                db.commit()
                db.refresh(rate)

db.close()




