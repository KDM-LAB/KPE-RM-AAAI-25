from scipy.stats import spearmanr, pearsonr, kendalltau
import numpy as np

def get_correlations(x, y):
    if len(x) < 3:
        return None # No point in computing correlation of 2 points. It'll either be +1, -1 NaN
    result = [list(pearsonr(x, y)), list(spearmanr(x, y)), list(kendalltau(x, y))]
    if np.any(np.isnan(result)):
        print(f"Found a correlation value of NaN, discarding it.\n{x=}\n{y=}")
        return None
    return {"Pearson": {"Correlation": result[0][0],"P-Value": result[0][1]},
            "Spearman": {"Correlation": result[1][0],"P-Value": result[1][1]},
            "Kendalltau": {"Correlation": result[2][0],"P-Value": result[2][1]}}


# -------------- Nishit's correlation function --------------

from itertools import permutations
from more_itertools import locate

def generate_permutations(lst):
    permutations_list = list(permutations(lst))
    return permutations_list

def find_indices(list_to_check, item_to_find):
    indices = locate(list_to_check, lambda x: x == item_to_find)
    return list(indices)

def generate_arrangements(input_list):       #To have all possible set of paper IDs in different arrangement
    arrangements = []
    current_arrangement = []

    def backtrack(index):
        if len(current_arrangement) == len(input_list):
            arrangements.append(current_arrangement[:])
            return

        element_dict = input_list[index]
        for value in element_dict.values():
            for element in value:
                current_arrangement.append(element)
                backtrack(index + 1)
                current_arrangement.pop()

    backtrack(0)
    return arrangements

def combination_papers(score):
    dict_orders={}
    dist_score=list(set(score))
    dist_score.sort()
    for i in range(len(dist_score)):
        index=find_indices(score,dist_score[i])
        dict_orders[i]=index
    
    dict_papers={}
    for i in dict_orders:
        temp=[]
        for k in dict_orders[i]:
            temp.append(score[k])
        dict_papers[i]=temp     #returns the ranked set of papers in ascending order
    
    #get permutations of all the elements in a list
    for i in dict_papers:
        if(len(dict_papers[i])!=1):
            aa=list(generate_permutations(dict_papers[i]))
            dict_papers[i]=aa   #permuted set of papers in ascending order
    
    list1=[]
    
    for i in range(len(dict_papers)):
        temp={}
        temp[i]=dict_papers[i]
        list1.append(temp)
    combos=generate_arrangements(list1)
    final_combination=[]
    for list_of_tuples in combos:
        final=[]
        for i in range(len(list_of_tuples)):
            if(type(list_of_tuples[i])==tuple):
                single_list = [item for item in list_of_tuples[i]]
                final.extend(single_list)
            else:
                final.append(list_of_tuples[i])
        final_combination.append(final)
    
    return final_combination

def get_correlations_sync(x, y):
    combination_original = combination_papers(x)
    combination_pke = combination_papers(y)

    pearson_coefficient=[]
    spearman_coefficient=[]
    kendaltau_coefficient=[]

    pearson_max_coeff=0
    spearman_max_coeff=0
    kendaltau_max_coeff=0

    for i in range(len(combination_original)):
        for j in range(len(combination_pke)):
            data1 = np.array(combination_original[i])
            data2 = np.array(combination_pke[j])

            p_coef, p = pearsonr(data1, data2)
            s_coef, p = spearmanr(data1, data2)
            k_coef, p = kendalltau(data1, data2)

            pearson_coefficient.append(p_coef)       
            spearman_coefficient.append(s_coef)       
            kendaltau_coefficient.append(k_coef)       
    try:
        pearson_max_coeff=max(pearson_coefficient)
    except:
        pearson_max_coeff=0
    
    try:
        spearman_max_coeff=max(spearman_coefficient)
    except:
        spearman_max_coeff=0

    try:
        kendaltau_max_coeff=max(kendaltau_coefficient)
    except:
        kendaltau_max_coeff=0

    return pearson_max_coeff, spearman_max_coeff, kendaltau_max_coeff
