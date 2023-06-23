from scipy.stats import spearmanr, pearsonr, kendalltau

def get_correlations(x, y):
    result = [list(pearsonr(x, y)), list(spearmanr(x, y)), list(kendalltau(x, y))]
    for r in result:
        for c in [0,1]:
            r[c] = round(r[c], 4)
    return {"Pearson": {"Correlation": result[0][0],"P-Value": result[0][1]},
            "Spearman": {"Correlation": result[1][0],"P-Value": result[1][1]},
            "Kendalltau": {"Correlation": result[2][0],"P-Value": result[2][1]}}



# Example
# a = [1,2,3,4,2,3,17]
# b = [4,2,5,0,4,5,15]

# print(pearsonr(a, b))

# print(spearmanr(a, b))
# print(pearsonr([1,2.5,4.5,6,2.5,4.5,7], \
#     [3.5,2,5.5,1,3.5,5.5,7]))

# print(kendalltau(a, b))
