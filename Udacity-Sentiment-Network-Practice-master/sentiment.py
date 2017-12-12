from collections import Counter
import numpy as np

POSITIVE = "POSITIVE"
NEGATIVE = "NEGATIVE"
layer_0 = np.zeros((1, 1))
word2index = {}

def pretty_print_review_and_label(reviews, labels, i):
    print (labels[i] + "\t:\t" + reviews[i][:80] + "...")

def read_and_label_data(dataset_file, labels_file):
    g = open(dataset_file,'r') # What we know!
    reviews = list(map(lambda x:x[:-1],g.readlines()))
    g.close()

    g = open(labels_file,'r') # What we WANT to know!
    labels = list(map(lambda x:x[:-1].upper(),g.readlines()))
    g.close()

    return reviews, labels

def calculate_word_ratio_in_reviews(reviews, labels):
    positive_counts = Counter()
    negative_counts = Counter()
    total_counts = Counter()
    pos_neg_ratios = Counter()

    # Count number of words in each type of review.
    for i in range(len(reviews)):
        words = reviews[i].split(" ")

        if labels[i] == POSITIVE:
            for word in words:
                positive_counts[word] += 1
                total_counts[word] += 1
        else:
            for word in words:
                negative_counts[word] += 1
                total_counts[word] += 1

    # Store positive/negative ratios for "common" words.
    for word, count in total_counts.most_common():
        # Consider words to be "common" if used at least 100 times.
        if count > 100:
            # Calculate the positive-to-negative ratio for a given word.
            # +1 in denominator ensures no division by zero for words that
            # only appear in positive reviews.
            pos_neg_ratio = positive_counts[word] / float(negative_counts[word] + 1)
            pos_neg_ratios[word] = np.log(pos_neg_ratio)

    return pos_neg_ratios, total_counts

def create_word_lookup_table(words):
    global word2index

    word2index = {}
    for i, word in enumerate(words):
        word2index[word] = i

def update_input_layer(review):
    global layer_0

    # Clear out previous state
    layer_0 *= 0

    words = review.split(" ")
    for word in words:
        layer_0[0, word2index[word]] += 1

# Convert a label to `0` or `1`.
def get_target_for_label(label):
    return 0 if label == NEGATIVE else 1

def main():
    global layer_0

    reviews, labels = read_and_label_data('reviews.txt', 'labels.txt')
    pos_neg_ratios, total_counts = calculate_word_ratio_in_reviews(reviews, labels)

    vocab = set(total_counts.elements())
    vocab_size = len(vocab)

    layer_0 = np.zeros((1, vocab_size))
    create_word_lookup_table(vocab)
    update_input_layer(reviews[0])

if __name__ == '__main__':
    main()
