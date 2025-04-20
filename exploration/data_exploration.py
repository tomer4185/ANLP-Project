import re
import matplotlib.pyplot as plt

from utils.preprocessing import get_parsed_data
# TODO: check about the weird token in the first song in get_data
data = get_parsed_data(20000)
# 1, number of words (Shaked)
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
plt.hist(verse_wc, bins=50, edgecolor='black')  # 'bins' controls number of bars
plt.xlabel("Word Count")
plt.ylabel("Frequency")
plt.title("Histogram of verses")
plt.savefig("../plots/verses_hist.png")

plt.hist(chorus_wc, bins=50, edgecolor='black')  # 'bins' controls number of bars
plt.xlabel("Word Count")
plt.ylabel("Frequency")
plt.title("Histogram of chorus")
plt.savefig("../plots/chorus_hist.png")

print(len(chorus_wc))

# 2. nuber of tokens (Shaked)
# 3. number of sentences/rows (Shaked)
# 4. sub-repetitions in the corus (Ariel)
# 5. repetitions of words in the paragraphs(yeah yeah yeah) (Ariel)
# 6. rhymes checking (Ariel)
# 7. number of unique words (Shaked)
# 8. precision and recall (Tomer)
# 9. single chorus in the song (Tomer)