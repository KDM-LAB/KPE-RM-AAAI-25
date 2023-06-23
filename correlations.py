from scipy.stats import spearmanr, pearsonr, kendalltau

def get_correlations(x, y):
    return pearsonr(x, y), spearmanr(x, y), kendalltau(x, y)



# Example
# a = [1,2,3,4,2,3,17]
# b = [4,2,5,0,4,5,15]

# print(pearsonr(a, b))

# print(spearmanr(a, b))
# print(pearsonr([1,2.5,4.5,6,2.5,4.5,7], \
#     [3.5,2,5.5,1,3.5,5.5,7]))

# print(kendalltau(a, b))
