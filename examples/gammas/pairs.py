Deltas = [-1.0, -0.5, 0.0, 0.5, 1.0]
gaps = [-0.5, 0.0, 0.5]
with open('pairs.txt', 'w') as f:
    for Delta in Deltas:
        for gap in gaps:
            f.write(f'{Delta} {gap}\n')