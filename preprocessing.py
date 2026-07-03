import re


def clean_review(review, english_stops):
    regex = re.compile(r'[^a-zA-Z\s]')
    review = regex.sub('', review)

    words = review.split(' ')
    filtered = [w for w in words if w not in english_stops]
    filtered = ' '.join(filtered)
    filtered = [filtered.lower()]

    return filtered
