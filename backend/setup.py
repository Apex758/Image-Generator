import nltk
import os

# Define the resources that need to be downloaded
NLTK_RESOURCES = [
    ('tokenizers/punkt', 'punkt'),
    ('taggers/averaged_perceptron_tagger', 'averaged_perceptron_tagger'),
    ('corpora/stopwords', 'stopwords')
]

def download_nltk_resources():
    """
    Downloads all necessary NLTK resources for the application.
    This script is intended to be run once during setup.
    """
    print("Starting NLTK resource download...")
    
    for resource_path, resource_name in NLTK_RESOURCES:
        try:
            # Check if the resource is already available
            nltk.data.find(resource_path)
            print(f"Resource '{resource_name}' already exists. Skipping.")
        except LookupError:
            # If not found, download it
            print(f"Downloading required NLTK resource: '{resource_name}'...")
            nltk.download(resource_name)
            print(f"Successfully downloaded '{resource_name}'.")

    print("\nAll required NLTK resources are available.")
    print("Setup complete.")

if __name__ == "__main__":
    download_nltk_resources()