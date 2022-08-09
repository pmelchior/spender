import argparse, pickle, torch, numpy as np
from spender.batch_wrapper import collect_batches

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("dir", help="data file directory")
    args = parser.parse_args()


    instrument_names = ["SDSS", "BOSS"]

    for name in instrument_names:
        files = collect_batches(args.dir, name)
        for filename in files:
            print (f"updating {filename}")
            with open(filename, 'rb') as f:
                spec, w, z, ids, norm, zerr = pickle.load(f)

                # remove tiny weights, set to zeros
                w[w<=1e-6] = 0

                # fix IDS: convert from string to (plate,mjd,fiberid)
                ids = torch.tensor(np.array(tuple(np.array(_.split("-"), dtype=int) for _ in ids)))

                batch = spec, w, z, ids, norm, zerr
            with open(filename, 'wb') as f:
                pickle.dump(batch, f)
