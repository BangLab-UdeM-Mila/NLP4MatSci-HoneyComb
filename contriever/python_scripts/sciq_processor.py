from base_dataset_processor import BaseDatasetProcessor
from datasets import load_dataset

class SCIQProcessor(BaseDatasetProcessor):
    def load_dataset(self):
        cache_dir = self.config['CACHE_DIR']
        self.dataset = load_dataset("allenai/sciq", cache_dir=cache_dir)
        self.dataset = self.dataset['train']

    def remove_empty_support_data(self):
        """Filters out any records without support given which is shown to be an empty string."""
        self.dataset = [data_entry for data_entry in self.dataset if data_entry['support'] != ""]


    def process_data(self):
        # Process support texts
        support_texts = [{'id': str(i), 'text': data_entry['support']} for i, data_entry in enumerate(self.dataset)]
        self.save_jsonl(support_texts, "support_text.jsonl", "Writing support text")

        # Process queries and answers
        data = [{'question': data_entry['question'], 'answers': [data_entry['support']]} for data_entry in self.dataset]
        self.save_jsonl(data, "data.jsonl", "Writing Q&A pairs")