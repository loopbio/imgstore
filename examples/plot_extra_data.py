import matplotlib.pyplot as plt

from imgstore import new_for_filename

store = new_for_filename('/tmp/DATA/20191231_085213')
assert store.has_extra_data

df = store.get_extra_data()
df.plot(subplots=True)

plt.show()
