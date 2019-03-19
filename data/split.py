
def main():
    path = 'clean/'
    splits = {'train': 0.8, 'eval': 0.2}
    files = ["sessions_161718_95014_top10.csv",
             "sessions_161718_95014_tail10.csv",
             "sessions_161718_top1000.csv"]
    for filename in files:
        split(path, filename, splits)


def split(path, filename, splits):
    lines = []
    # assert len(splits) == 2
    with open(path + filename, 'r') as f:
        lines = f.readlines()
    header = lines[0]
    lines = lines[1:]
    assert len(lines) >= len(splits)

    print(filename, len(lines))

    start_lines = 0
    for i, (name, frac) in enumerate(splits.items()):
        filename_split = "{}_{}".format(name, filename)
        num_lines = int(frac*len(lines))
        end_lines = start_lines + num_lines
        if i == len(splits) - 1:
            lines_out = [header] + lines[start_lines:]
        else:
            lines_out = [header] + lines[start_lines:end_lines]
        with open(path + filename_split, 'w') as f_out:
            f_out.writelines(lines_out)
        start_lines = end_lines


if __name__ == "__main__":
    main()

