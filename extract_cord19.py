import csv
import json

final = []

# open the file
with open('metadata.csv') as f_in:
    reader = csv.DictReader(f_in)
    for row in reader:
        # access metadata
        title = row['title']
        abstract = row['abstract']
        text = []
        if row['pmc_json_files']:
            for json_path in row['pmc_json_files'].split('; '):
                with open(json_path) as f_json:
                    full_text_dict = json.load(f_json)
                    # grab full text
                    for paragraph_dict in full_text_dict['body_text']:
                        paragraph_text: str = paragraph_dict['text']
                        if len(paragraph_text) > 1:
                            text.append(paragraph_text)
        if len(text) == 0 and len(abstract) < 2:
            continue
        final.append({
            'title': title,
            'abstract': abstract,
            'text': text
        })
with open("data/cord19-filtered.json", 'w') as f_json:
    json.dump(final, f_json)