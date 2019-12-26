
def nocall_check(line):
    threshold = 10 #max number of nocall per position

    if int(line[2]) < threshold: #the third column is the number of sample with nocall
        return True
    else:
        return False


if __name__ == "__main__":
    from pathlib import Path
    dir = Path('/d/data/plasmo/nat_out')
    nocall_path = dir / 'nocall.tsv'
    table_path = dir / 'variants.tsv'
    out_path = dir / 'nocall_filtered.tsv'

    with open(nocall_path, 'r') as input:
        nocall_lines = [l.split('\t') for l in input]
    with open(table_path, 'r') as input:
        table_lines = [l for l in input]
    
    good = [tl for tl, nl in zip(table_lines[1:], nocall_lines[1:]) if nocall_check(nl)]

    with open(out_path, 'w') as output:
        output.write(table_lines[0] + '\n')
        output.write('\n'.join(good))

    print('{0} good positions out of {1} for {2} percent'.format(len(good), len(table_lines), round(len(good) / len(table_lines) * 100, 2)))