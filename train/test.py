from spender.data.sdss import SDSS, BOSS

dir = "/Users/pmelchior/data"
instrument = SDSS()

data_loader = instrument.get_data_loader(dir, which="train", shuffle=False)
print(next(iter(data_loader))[2])

print(instrument.get_spectrum(dir, 273, 51957, 1))

ids = [[273, 51957, 1], [578, 52339, 424]]
print(instrument.make_batch(dir, ids))
