from collections import defaultdict
from operator import itemgetter
from matplotlib import pyplot as plt

distribution = defaultdict(int)

with open( './data/store.txt', 'r', encoding = 'utf-8' ) as store:
    while True:
        line = store.readline()

        if not line:
            break

        try:
            _, headline, _ = line.split( '|' )
        except Exception as e:
            print( line )
            raise e

        for c in headline:
            distribution[c] += 1

sorted_dist = sorted( distribution.items(), key=itemgetter( 1 ) )

with open( './data/distribution.txt', 'w', encoding = 'utf-8' ) as dist_file:
    for char, occurence in sorted_dist:
        dist_file.write( str(char) + ', ' + str(occurence) + '\n' )
    dist_file.write( '"' + '", "'.join( sorted( [char for char, _ in sorted_dist] ) ) + '"\n' )

plt.hist( [o for c, o in distribution.items()], label = [c for c, o in distribution.items()] )
plt.show()