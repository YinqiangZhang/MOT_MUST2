import numpy as np
import os
import re

root = './results/'
results_set = []
match_str1 = re.compile('MOT16-[0-9]{2}\.txt')
match_str2 = re.compile('b[0-9]\.txt')

if not os.path.exists('./modified_results_MOT4/'):
    os.makedirs('./modified_results_MOT4/')

for filename in os.listdir(root):
    if re.match(match_str2, filename):
        results_set.append(filename)

for filename in results_set:

    results_path = os.path.join('./results/', filename)
    results = np.loadtxt(results_path, dtype=float, delimiter=',')
    results = results[np.argsort(results[:, 0])]
    id_num = len(np.unique(results[:, 1]))
    print(id_num)

    np.savetxt('./modified_results_MOT4/'+filename, results[:, :6], fmt='%d,%d,%.1f,%.1f,%.1f,%.1f')
