__MAPPING__ = {
    'A': 0,
    'N': 1,
    'O': 2,
    '~': 3
}

__REVERSE_MAPPING__ = {
    0: 'A',
    1: 'N',
    2: 'O',
    3: '~'
}


def format_labels(labels):
    return [__MAPPING__[x] for x in labels]


def get_original_label(category):
    return __REVERSE_MAPPING__[category]
