from collections import defaultdict
from operator import itemgetter
from matplotlib import pyplot as plt

distribution = defaultdict(int)

with open( './data/headlines.csv', 'r', encoding = 'utf-8' ) as store:
    while True:
        line = store.readline()

        if not line:
            break

        for c in line:
            distribution[c] += 1

sorted_dist = sorted( distribution.items(), key=itemgetter( 1 ) )

with open( './data/distribution.txt', 'w', encoding = 'utf-8' ) as dist_file:
    print( str(len(sorted_dist)) + " chars" )
    
    for char, occurence in sorted_dist:
        dist_file.write( str(char) + ', ' + str(occurence) + '\n' )

    chars = sorted( [char for char, _ in sorted_dist] )

    if '\n' in chars:
        chars[chars.index('\n')] = "\\n"
    if '"' in chars:
        chars[chars.index('"')] = "\\\""

    dist_file.write( '"' + '", "'.join( chars ) + '"\n' )

plt.hist( [o for c, o in distribution.items()], label = [c for c, o in distribution.items()] )
plt.show()