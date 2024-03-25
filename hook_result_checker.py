import numpy as np
from numpy.linalg import norm
import os
import argparse
from tqdm import tqdm


def check(pathA, pathB):
    statistic = {}
    title = None
    assert os.path.isdir(pathA), f'[Checker] {pathA} is not exist or not a folder.'
    assert os.path.isdir(pathB), f'[Checker] {pathB} is not exist or not a folder.'
    for rank_folder in sorted(os.listdir(pathA)):
        print(f'[Checker] Checking for rank {rank_folder}.')
        if os.path.isdir(f'{pathA}/{rank_folder}') and os.path.isdir(f'{pathB}/{rank_folder}'):
            assert rank_folder not in statistic.keys(), f"[Checker] Encounter a repeat rank {rank_folder}."
            statistic[rank_folder] = {}
            tensors = os.listdir(f'{pathA}/{rank_folder}')
            tensors = sorted(tensors, key=lambda x: int(x.split('_')[2]))
            for tensor in tqdm(tensors):
                if not os.path.exists(f'{pathB}/{rank_folder}/{tensor}'):
                    print(f'[Checker] Will skip checking tensor {tensor} for rank {rank_folder} '
                          f'since it exists in {pathA} but not exists in {pathB}.')
                    continue
                a = np.load(f'{pathA}/{rank_folder}/{tensor}').astype(np.float32).flatten()
                b = np.load(f'{pathB}/{rank_folder}/{tensor}').astype(np.float32).flatten()
                meanA = np.mean(a)
                meanB = np.mean(b)
                stdA = np.std(a)
                stdB = np.std(b)
                varA = np.var(a)
                varB = np.var(b)
                statistic[rank_folder][tensor] = {
                    'cosine similarity': np.dot(a, b) / (norm(a) * norm(b)),
                    'meanA': meanA,
                    'meanB': meanB,
                    'meanA - meanB': meanA - meanB,
                    'stdA': stdA,
                    'stdB': stdB,
                    'stdA - stdB': stdA - stdB,
                    'varA': varA,
                    'varB': varB,
                    'varA - varB': varA - varB
                }
                if title is None:
                    title = 'tensor_name'
                    for name in statistic[rank_folder][tensor].keys():
                        title += f',{name}'
                    title += '\n'

    os.makedirs(f'{pathA}/compare_rst', exist_ok=True)
    for rank in statistic.keys():
        print(f'[Checker] Saving rst for rank {rank}')
        results = statistic[rank]
        tensors = sorted(results.keys(), key=lambda x: int(x.split('_')[2]))
        with open(f'{pathA}/compare_rst/{rank}_rst.csv', 'w') as file:
            file.write(title)
            for tensor in tensors:
                rst = f'{tensor}'
                for k, v in results[tensor].items():
                    rst += f',{v}'
                rst += '\n'
                file.write(rst)
    print(f'[Checker] Checking results have been saved at {pathA}/compare_rst.')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--pathA', type=str)
    parser.add_argument('--pathB', type=str)
    args = parser.parse_args()
    pathA = args.pathA
    pathB = args.pathB
    assert pathA is not None and pathB is not None, "Should provide --pathA and --pathB"
    print(f'[Checker] Comparing {pathA}/tensors with {pathB}/tensors.')
    check(f'{pathA}/tensors', f'{pathB}/tensors')
    print(f'[Checker] Comparing {pathA}/grads with {pathB}/grads.')
    check(f'{pathA}/grads', f'{pathB}/grads')


if __name__ == '__main__':
    main()
