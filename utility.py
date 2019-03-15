def extract_feature_from_tree(tree, feature_name):
    feature = tree.tree_.feature
    feature_set = set()
    for i in feature:
        if i == -2:
            continue
        if feature_name[i] not in feature_set:
            feature_set.add(feature_name[i])
    return feature_set
