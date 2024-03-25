import argparse
from matplotlib import pyplot as plt

name_to_idx = {
    'cosine similarity': 1,
    'meanA': 2,
    'meanB': 3,
    'meanA - meanB': 4,
    'stdA': 5,
    'stdB': 6,
    'stdA - stdB': 7,
    'varA': 8,
    'varB': 9,
    'varA - varB': 10
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--rst_path', type=str)
    parser.add_argument('--target_name', type=str)
    parser.add_argument('--values_per_step', type=str)
    args = parser.parse_args()
    rst_path = args.rst_path
    target_name = args.target_name
    values_per_step = args.values_per_step
    assert rst_path is not None, "Should provide --rst_path"
    assert target_name is not None, "Should provide --target_name"
    idx = []
    count = 0
    values = []
    with open(rst_path, 'r') as f:
        for line in f:
            line = line.strip()
            if 'tensor_name' in line:
                continue
            else:
                line = line.split(',')
                idx.append(count)
                count += 1
                values.append(float(line[name_to_idx[target_name]]))

    if values_per_step is not None:
        values_per_step = int(values_per_step)
        for x in range(0, len(values), values_per_step):
            plt.axvline(x, color='r', linestyle='dotted')
    plt.plot(idx, values, label=target_name)
    plt.legend()
    plt.show()


if __name__ == '__main__':
    main()
