import pandas as pd
import re

def parse_lyrics_sections(lyrics):
    # Define valid section labels with regex support
    valid_labels = [
        'intro',
        'verse',
        'verse\\s*\\d*',
        'pre[-\\s]?chorus',
        'chorus',
        'bridge',
        'outro',
        'refrain',
        'post[-\\s]?chorus'
    ]
    joined_labels = '|'.join(valid_labels)

    # Pattern: line must contain ONLY the tag, optional brackets, case-insensitive
    section_header_pattern = re.compile(
        rf'^\s*[\[\(\{{]?\s*({joined_labels})\s*[\]\)\}}]?\s*$',
        flags=re.IGNORECASE
    )

    lines = lyrics.strip().splitlines()

    # First line must be a section label
    if not lines or not section_header_pattern.match(lines[0]):
        return None

    result = {}
    current_section = None
    current_lines = []
    saw_chorus = False

    for line in lines:
        match = section_header_pattern.match(line)
        if match:
            # Save previous section
            if current_section and current_lines:
                part_text = '\n'.join(current_lines).strip()
                if part_text:
                    result[part_text] = current_section
                current_lines = []

            # Normalize tag (remove spaces and dashes)
            tag = match.group(1).lower().replace(' ', '').replace('-', '')
            if tag in ['chorus', 'prechorus', 'postchorus']:
                current_section = tag  # keep as-is
                saw_chorus = True
            elif 'verse' in tag:
                current_section = 'verse'
            elif tag in ['intro', 'bridge', 'outro', 'refrain']:
                current_section = tag
            else:
                current_section = None  # shouldn't happen
        elif current_section:
            current_lines.append(line)

    # Save the final section
    if current_section and current_lines:
        part_text = '\n'.join(current_lines).strip()
        if part_text:
            result[part_text] = current_section

    return result if saw_chorus and result else None

def get_parsed_data():
    df = pd.read_parquet("hf://datasets/mrYou/Lyrics_eng_dataset/data/train-00000-of-00001.parquet")
    parsed_data = {}
    for i, row in enumerate(df.head(30).iterrows()):
        lyrics = row[1]["lyrics"]
        parsed_row = parse_lyrics_sections(lyrics)
        if parsed_row is not None:
            parsed_data[row[1]["id"]] = parsed_row
    return parsed_data

def prepare_data_for_training(parsed_data):
    # This function should prepare the parsed data for training
    # For example, you might want to convert it into a format suitable for your model
    pass

if __name__ == "__main__":
    parsed_data = get_parsed_data()
    print("data parsed")
