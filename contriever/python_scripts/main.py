import os

from sciq_processor import SCIQProcessor

def main():
    config = {
        'SAVING_DIRECTORY': os.environ["SAVING_DIRECTORY"],
        'CACHE_DIR': os.environ['TRANSFORMERS_CACHE'],
        'OUTPUT_DIR': os.path.join(os.environ["SAVING_DIRECTORY"], "output")
    }
    sciq_processor = SCIQProcessor(config)
    sciq_processor.load_dataset()
    sciq_processor.remove_empty_support_data()
    sciq_processor.process_data()

if __name__ == "__main__":
    main()
