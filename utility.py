import pickle as pl


def extract_feature_from_tree(tree, feature_name):
    feature = tree.tree_.feature
    feature_set = set()
    for i in feature:
        if i == -2:
            continue
        feature_set.add(feature_name[i])
    return feature_set


def store_model(file_name, model):
    with open(file_name, 'wb') as f:
        pl.dump(model, f)


def read_model(file_name):
    with open(file_name, 'rb') as f:
        model = pl.load(f)
        return model

