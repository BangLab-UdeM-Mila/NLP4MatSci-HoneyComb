from contriever_framework.python_scripts.base_dataset_processor import BaseDatasetProcessor


class DataProcessor(BaseDatasetProcessor):
    def load_dataset(self, question, recalled_documents):
        self.question = question
        self.recalled_documents = recalled_documents

    def process_data(self, batch_n=None):
        # Process support texts
        support_texts = [{'id': str(i), 'text': document} for i, document in enumerate(self.recalled_documents)]
        self.save_jsonl(support_texts, f"support_text_{batch_n}.jsonl", "Writing support text")

        # Process queries and answers
        data = [{'question': self.question, 'answers': []}]
        self.save_jsonl(data, f"data_{batch_n}.jsonl", "Writing Q&A pairs")