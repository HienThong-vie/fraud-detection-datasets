"""Portable native-column transformation for saved CatBoost pipelines."""
def native_frame(frame, columns, categorical):
    prepared = frame[columns].copy()
    for column in categorical:
        prepared[column] = prepared[column].astype('string').fillna('__MISSING__').astype(str)
    return prepared
