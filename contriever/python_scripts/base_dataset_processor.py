import os
import json
import logging
from abc import ABC, abstractmethod

import pandas as pd
from tqdm import tqdm

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class BaseDatasetProcessor(ABC):
    """Base class for dataset processing."""
    def __init__(self, config):
        self.config = config
        self.dataset = None
        self.setup_directories()

    def setup_directories(self):
        os.makedirs(self.config['SAVING_DIRECTORY'], exist_ok=True)
        os.makedirs(self.config['OUTPUT_DIR'], exist_ok=True)

    @abstractmethod
    def load_dataset(self):
        """Load dataset from source."""
        pass

    @abstractmethod
    def process_data(self):
        """Process data to generate required output."""
        pass

    def save_jsonl(self, data, filename, desc="Saving data"):
        path = os.path.join(self.config['OUTPUT_DIR'], filename)
        with open(path, 'w') as f:
            for item in tqdm(data, desc=desc):
                json.dump(item, f)
                f.write('\n')
        logging.info(f"Data written to {path}")

