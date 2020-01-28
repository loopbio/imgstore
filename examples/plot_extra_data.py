import sys

import matplotlib.pyplot as plt

from imgstore import new_for_filename

try:
    SOURCE = sys.argv[1]
except IndexError:
    SOURCE = '/mnt/loopbio/tests/motif/mcc8_20191231_155718/'

store = new_for_filename(SOURCE)
assert store.has_extra_data

df = store.get_extra_data(ignore_corrupt_chunks=True)
df.plot(subplots=True)

plt.show()
