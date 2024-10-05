import os
import json

import numpy as np
import pandas as pd
from rank_bm25 import BM25Okapi


SAVING_DIRECTORY = os.environ["SAVING_DIRECTORY"]


class KnowledgeBase:
    def __init__(self, filepath, logger=None):
        self.logger = logger
        self.filepath = filepath
        self.documents = self.load()
        self.bm25 = BM25Okapi([doc.split(" ") for doc in self.documents])
        
        self.logger.info(f"loadded documents from {filepath} into a {type(self.documents)}")
        self.logger.info(f"containing {len(self.documents)} elements") if isinstance(self.documents, list) else None

    def load(self):
        with open(self.filepath, 'r', encoding='utf8') as file:
            knowledge_base = json.load(file)['Material Science']['Children']

        document_list = []

        for k1 in knowledge_base.keys():
            tmp = knowledge_base[k1]['Children']
            for k2 in tmp.keys():
                title = tmp[k2]['title']
                content = tmp[k2]['content']
                document_list.append(f"{title} {content}")
        
        sciq_knowledge = pd.read_parquet(f"{SAVING_DIRECTORY}/dataset/sciq/sciq.parquet")

        sciqa_docs = list(set(sciq_knowledge.support))[1:]

        sciqa_docs.extend(document_list)
        
        return sciqa_docs
    
    def search(self, query, n_recall=5):
        tokenized_query = query.split(" ")
        doc_scores = self.bm25.get_scores(tokenized_query)
        top_indexes = np.flip(np.argsort(doc_scores)[-n_recall:])
        top_documents = [self.documents[i] for i in top_indexes]
        return top_documents
