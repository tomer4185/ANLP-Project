import re
import matplotlib.pyplot as plt

from utils.preprocessing import get_parsed_data

# TODO: check about the weird token in the first song in get_data
data = get_parsed_data(20000)

# 1, number of words (Shaked)
def number_of_words_plot(data):
    verse_wc = []
    chorus_wc = []
    for key in data.keys():
        song = data[key]
        for part in song:
            part_split = re.split(r'[ \n]',  part)
            if song[part] == "verse":
                verse_wc.append(len(part_split))
            elif song[part] == "chorus":
                chorus_wc.append(len(part_split))
            else:
                continue

    print(len(verse_wc))
    print(len(chorus_wc))
    plt.hist(verse_wc, bins=50, color='blue', alpha=0.6, label='verse', edgecolor='black', density=True)
    plt.hist(chorus_wc, bins=50, color='orange', alpha=0.6, label='chorus', edgecolor='black', density=True)
    plt.xlabel("Amount of Word in Part")
    plt.ylabel("Frequency")
    plt.title("Histogram of verses")
    plt.legend()
    plt.savefig("../plots/word_count_hist.png")
    # plt.show()


def number_of_row_plot(data):
    verse_rows_count = []
    chorus_rows_count = []
    for key in data.keys():
        song = data[key]
        for part in song:
            part_split = re.split(r'[\n]',  part)
            if song[part] == "verse":
                verse_rows_count.append(len(part_split))
            elif song[part] == "chorus":
                chorus_rows_count.append(len(part_split))
            else:
                continue

    print(len(verse_rows_count))
    print(len(chorus_rows_count))
    plt.hist(verse_rows_count, bins=50, color='blue', alpha=0.6, label='verse', edgecolor='black', density=True)
    plt.hist(chorus_rows_count, bins=50, color='orange', alpha=0.6, label='chorus', edgecolor='black', density=True)
    plt.xlabel("Amount of Rows in Part")
    plt.ylabel("Frequency")
    plt.title("Histogram of the Rows in Verses")
    plt.legend()
    plt.savefig("../plots/rows_count_hist.png")
    # plt.show()

# number_of_words_plot(data)
number_of_row_plot(data)

# 2. nuber of tokens (Shaked)

# 3. number of sentences/rows (Shaked)
# 4. sub-repetitions in the corus (Ariel)
# 5. repetitions of words in the paragraphs(yeah yeah yeah) (Ariel)
# 6. rhymes checking (Ariel)
# 7. number of unique words (Shaked)
# 8. precision and recall (Tomer)
# 9. single chorus in the song (Tomer)