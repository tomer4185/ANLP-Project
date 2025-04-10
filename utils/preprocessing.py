import pandas as pd
import re

LABELS= {"verse": 0, "chorus": 1}

def parse_lyrics_sections(lyrics):
    """
    parses the lyrics of the song into sections (chorus, verse, bridge, etc.)
    :param lyrics: the whole lyrics of the song
    :return: the lyrics parsed, or None if the song parts are not mentioned or do not include a chorus
    """
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

def get_parsed_data() -> dict:
    """
    fetches parsed data from the database.
    returns a dict of ficts,
    where the keys are the ids of the songs, and the values are dicts, one per song,
    whose keys are the lyrics of a part of the song, and its label (verse, chorus, bridge, etc.)
    """
    df = pd.read_parquet("hf://datasets/mrYou/Lyrics_eng_dataset/data/train-00000-of-00001.parquet")
    parsed_data = {}
    for i, row in enumerate(df.head(1000).iterrows()):
        lyrics = row[1]["lyrics"]
        parsed_row = parse_lyrics_sections(lyrics)
        if parsed_row is not None:
            parsed_data[row[1]["id"]] = parsed_row
    return parsed_data

def build_format_text(part_text, section_context):
    # build the formated text
    # the form for each part should be in the form of [CLS] current_part_text [SEP] full_song_context [SEP]
    formatted_text = f"[CLS] {part_text} [SEP] {section_context} [SEP]"
    return formatted_text

def prepare_data_for_training(parsed_data):
    """
    Build the data to train the model
    The parser neglects parts-of-song that are not chorus or verse
    The form for each part should be in the form of [CLS] current_part_text [SEP] full_song_context [SEP]
    :param parsed_data: the whole data in the format of dict of dicts
    :return: a pandas df where each row represents a part of a song,
    and contains as text the lyrics in the format of
    [CLS] {part_text} [SEP] {section_context} [SEP], where the section_context is all the other parts of songs, concatenated
    the label is 0 if the part_text is verse and 0 if chorus
    """

    data = []
    for song_id, sections in parsed_data.items():
        for i, (part_text, section) in enumerate(sections.items()):
            if section not in LABELS.keys():
                continue
            # get section context
            section_context = [text for text in sections if text != part_text]
            # todo: dont we want to randomize it?
            # todo: leave only one chorus per song?
            if len (section_context) < 2:
                continue
            section_context = " ".join(section_context)
            # build the data
            format_text = build_format_text(part_text, section_context)
            data.append((format_text, LABELS[section]))
    return pd.DataFrame(data, columns=["text", "label"])


def get_data():
    """
    check out prepare_data_for_training
    """
    parsed_data = get_parsed_data()
    data = prepare_data_for_training(parsed_data)
    print("data parsed")
    return data
