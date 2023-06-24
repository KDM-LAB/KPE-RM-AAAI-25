from db import Base
from sqlalchemy import Boolean, Column, ForeignKey, Integer, String, Float, DateTime
from sqlalchemy.orm import relationship
from sqlalchemy.schema import PrimaryKeyConstraint, ForeignKeyConstraint
import datetime

class Reviewers(Base):
    __tablename__ = "reviewers"

    reviewer_pk = Column(Integer, primary_key=True)#, index=True) # autoincrement = True
    author_id = Column(String(100), unique=True)#, index=True)
    name = Column(String(100))

    def __repr__(self):
        return f"reviewer_pk: {self.reviewer_pk} || author_id: {self.author_id}"

class Papers(Base):
    __tablename__ = "papers"

    paper_pk = Column(Integer, primary_key=True)#, index=True) # autoincrement = True
    ssId = Column(String(100))#, unique=True)#, index=True) # they will have 24 duplicates
    title = Column(String(1000), default=None)
    abstract = Column(String(10000), default=None)
    pdf_text_path = Column(String(100), default=None) # Since PDF text is not getting stored even with Text/BLOB, I am storing only the path
    year = Column(Integer)
    is_submitted = Column(Boolean)

    def __repr__(self):
        return f"paper_pk: {self.paper_pk} || ssId: {self.ssId}"

class Reviewers_Papers(Base):
    __tablename__ = "reviewers_papers"

    reviewer_pk = Column(Integer, ForeignKey('reviewers.reviewer_pk'), primary_key=True)#, index=True) # autoincrement = True
    paper_pk = Column(Integer, ForeignKey('papers.paper_pk'), primary_key=True)#, index=True) # autoincrement = True

    # rp_review = relationship("Reviewers", backref="reviewers_papers") # will do it again
    # rp_paper = relationship("Papers", backref="reviewers_papers")

    __table_args__ = (PrimaryKeyConstraint('reviewer_pk', 'paper_pk'),) # beware of comma! from chat gpt

    def __repr__(self):
        return f"reviewer_pk: {self.reviewer_pk} || paper_pk: {self.paper_pk}"

class Rating(Base):
    __tablename__ = "rating"

    reviewer_pk = Column(Integer, ForeignKey('reviewers.reviewer_pk'), primary_key=True)#, index=True) # autoincrement = True
    paper_pk = Column(Integer, ForeignKey('papers.paper_pk'), primary_key=True)#, index=True) # autoincrement = True
    rating = Column(Float)

    # rp_review = relationship("Reviewers", back_populates="reviewers_papers") # will do it again
    # rp_paper = relationship("Papers", back_populates="reviewers_papers")

    __table_args__ = (PrimaryKeyConstraint('reviewer_pk', 'paper_pk'),) # beware of comma! from chat gpt

    def __repr__(self):
        return f"reviewer_pk: {self.reviewer_pk} || paper_pk: {self.paper_pk} || rating: {self.rating}"

class Model_Paper_Keywords(Base):
    __tablename__ = "model_paper_keywords"

    pk = Column(Integer, primary_key=True)#, index=True) # autoincrement = True
    paper_pk = Column(Integer, ForeignKey('papers.paper_pk'))#, index=True) # autoincrement = True
    model_name = Column(String(100))
    model_keywords_wo_pdf = Column(String(1000), default=None) # For all the papers, only abstract and title are used
    model_keywords_w_pdf = Column(String(1000), default=None) # If a paper doesn't have pdf_text then it's abstract and title will be used (if available)
    timestamp = Column(DateTime, default=datetime.datetime.utcnow) # 5:30 hours delay wrt India

    def __repr__(self):
        return f"paper_pk: {self.paper_pk} || model_name: {self.model_name}"
        
class Model_Reviewer_Paper_Similarity(Base): # as of now this is only aimed for entries which are in ratings table. No other permutation since that data is not available.
    __tablename__ = "model_reviewer_paper_similarity"

    pk = Column(Integer, primary_key=True)#, index=True) # autoincrement = True
    reviewer_pk = Column(Integer, ForeignKey('rating.reviewer_pk'))#, index=True) # autoincrement = True
    paper_pk = Column(Integer, ForeignKey('rating.paper_pk'))#, index=True) # autoincrement = True
    model_name = Column(String(100))
    model_similarity_wo_pdf = Column(Float)
    model_similarity_w_pdf = Column(Float)
    timestamp = Column(DateTime, default=datetime.datetime.utcnow) # 5:30 hours delay wrt India

    __table_args__ = (ForeignKeyConstraint([reviewer_pk, paper_pk], [Rating.reviewer_pk, Rating.paper_pk]),)

    def __repr__(self):
        return f"reviewer_pk: {self.reviewer_pk} || paper_pk: {self.paper_pk} || model_similarity: {self.model_similarity}"
        














# class User(Base):
#     __tablename__ = "users"

#     user_id = Column(Integer, primary_key=True, index=True) # autoincrement = True
#     email = Column(String(100), unique=True, index=True)
#     hashed_password = Column(String(100))
#     is_active = Column(Boolean, default=True)

#     items = relationship("Item", back_populates="owner")

#     def __repr__(self):
#         return f"user_id: {self.user_id} || user_email: {self.email}"

# class Item(Base):
#     __tablename__ = "items"

#     item_id = Column(Integer, primary_key=True, index=True)
#     title = Column(String(50), index=True)
#     description = Column(String(200), index=True)
#     owner_id = Column(Integer, ForeignKey("users.user_id"))

#     owner = relationship("User", back_populates="items")

#     def __repr__(self):
#         return f"item_id: {self.item_id} || item_title: {self.title}"
