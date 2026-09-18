#!/usr/bin/env python

import argparse
import os
import time

import numpy as np
import torch
from accelerate import Accelerator
from spender import SpectrumAutoencoder
from spender.data.sdss import SDSS
from spender.loss import LossConfig, get_losses
from spender.util import mem_report

# allows one to run fp16_train.py from home directory
import sys;sys.path.insert(1, './')


def prepare_train(seq,niter=1000):
    for d in seq:
        if not "iteration" in d:d["iteration"]=niter
        if not "encoder" in d:d.update({"encoder":d["data"]})
    return seq

def build_ladder(train_sequence):
    n_iter = sum([item['iteration'] for item in train_sequence])

    ladder = np.zeros(n_iter,dtype='int')
    n_start = 0
    for i,mode in enumerate(train_sequence):
        n_end = n_start+mode['iteration']
        ladder[n_start:n_end]= i
        n_start = n_end
    return ladder

def get_all_parameters(models,instruments):
    model_params = []
    # multiple encoders
    for model in models:
        model_params += model.encoder.parameters()
    # 1 decoder
    model_params += model.decoder.parameters()
    dicts = [{'params':model_params}]

    n_parameters = sum([p.numel() for p in model_params if p.requires_grad])

    instr_params = []
    # instruments
    for inst in instruments:
        if inst==None:continue
        instr_params += inst.parameters()
        s = [p.numel() for p in inst.parameters()]
    if instr_params != []:
        dicts.append({'params':instr_params,'lr': 1e-4})
        n_parameters += sum([p.numel() for p in instr_params if p.requires_grad])
        print("parameter dict:",dicts[1])
    return dicts,n_parameters

# per-instrument hyperparameters of the similarity and consistency losses
# (currently identical for both instruments, kept separate for future tuning)
DEFAULT_LOSS_CONFIGS = {
    "SDSS": LossConfig(consistency_tol=0.1, restframe_mu=5000, restframe_sigma=1000, similarity_data_amp=20),
    "BOSS": LossConfig(consistency_tol=0.1, restframe_mu=5000, restframe_sigma=1000, similarity_data_amp=20),
}

def checkpoint(accelerator, args, optimizer, scheduler, n_encoder, outfile, losses):
    unwrapped = [accelerator.unwrap_model(args_i).state_dict() for args_i in args]

    accelerator.save({
        "model": unwrapped,
        "losses": losses,
    }, outfile)
    return

def load_model(filename, models, instruments):
    device = instruments[0].wave_obs.device
    model_struct = torch.load(filename, map_location=device)

    for i, model in enumerate(models):
        # backwards compat: encoder.mlp instead of encoder.mlp.mlp
        if 'encoder.mlp.mlp.0.weight' in model_struct['model'][i].keys():
            from collections import OrderedDict
            model_struct['model'][i] = OrderedDict([(k.replace('mlp.mlp', 'mlp'), v) for k, v in model_struct['model'][i].items()])
        # backwards compat: add instrument to encoder
        try:
            model.load_state_dict(model_struct['model'][i], strict=False)
        except RuntimeError:
            model_struct['model'][i]['encoder.instrument.wave_obs']= instruments[i].wave_obs
            model_struct['model'][i]['encoder.instrument.skyline_mask']= instruments[i].skyline_mask
            model.load_state_dict(model_struct[i]['model'], strict=False)

    losses = model_struct['losses']
    return models, losses


def train(models,
          instruments,
          trainloaders,
          validloaders,
          n_epoch=200,
          outfile=None,
          losses=None,
          verbose=False,
          lr=1e-4,
          n_batch=50,
          aug_fcts=None,
          similarity=True,
          consistency=True,
          loss_configs=None,
          ):

    n_encoder = len(models)
    model_parameters, n_parameters = get_all_parameters(models,instruments)

    if verbose:
        print("model parameters:", n_parameters)
        mem_report()

    ladder = build_ladder(train_sequence)
    optimizer = torch.optim.Adam(model_parameters, lr=lr, eps=1e-4)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, lr,
                                              total_steps=n_epoch)

    accelerator = Accelerator(mixed_precision='fp16')
    models = [accelerator.prepare(model) for model in models]
    instruments = [accelerator.prepare(instrument) for instrument in instruments]
    trainloaders = [accelerator.prepare(loader) for loader in trainloaders]
    validloaders = [accelerator.prepare(loader) for loader in validloaders]
    optimizer = accelerator.prepare(optimizer)
    if loss_configs is None:
        loss_configs = [DEFAULT_LOSS_CONFIGS[instrument.__class__.__name__] for instrument in instruments]

    # define losses to track
    n_loss = 3
    epoch = 0
    if losses is None:
        detailed_loss = np.zeros((2, n_encoder, n_epoch, n_loss))
    else:
        try:
            epoch = len(losses[0][0])
            n_epoch += epoch
            detailed_loss = np.zeros((2, n_encoder, n_epoch, n_loss))
            detailed_loss[:, :, :epoch, :] = losses
            if verbose:
                losses = tuple(detailed_loss[0, :, epoch-1, :])
                vlosses = tuple(detailed_loss[1, :, epoch-1, :])
                print(f'====> Epoch: {epoch-1}')
                print('TRAINING Losses:', losses)
                print('VALIDATION Losses:', vlosses)
        except: # OK if losses are empty
            pass

    if outfile is None:
        outfile = "checkpoint.pt"

    for epoch_ in range(epoch, n_epoch):

        mode = train_sequence[ladder[epoch_ - epoch]]

        # turn on/off model decoder
        for p in models[0].decoder.parameters():
            p.requires_grad = mode['decoder']

        slope = ANNEAL_SCHEDULE[(epoch_ - epoch)%len(ANNEAL_SCHEDULE)]
        if n_epoch-epoch_<=10: slope=0 # turn off similarity
        for config in loss_configs:
            config.similarity_slope = slope

        if verbose and similarity:
            print("similarity info:",slope)

        for which in range(n_encoder):

            # turn on/off encoder
            for p in models[which].encoder.parameters():
                p.requires_grad = mode['encoder'][which]

            # optional: training on single dataset
            if not mode['data'][which]:
                continue

            models[which].train()
            instruments[which].train()

            n_sample = 0
            for k, batch in enumerate(trainloaders[which]):
                batch_size = len(batch[0])
                losses = get_losses(
                    models[which],
                    instruments[which],
                    batch,
                    aug_fct=aug_fcts[which],
                    similarity=similarity,
                    consistency=consistency,
                    loss_config=loss_configs[which],
                )
                # weighted combination of the individual losses for backprop
                fidelity_loss, sim_loss, cons_loss = losses
                config = loss_configs[which]
                loss = fidelity_loss + config.similarity_amp * sim_loss + config.consistency_amp * cons_loss
                accelerator.backward(loss)
                # clip gradients: stabilizes training with similarity
                accelerator.clip_grad_norm_(model_parameters[0]['params'], 1.0)
                # once per batch
                optimizer.step()
                optimizer.zero_grad()

                # logging: training
                detailed_loss[0][which][epoch_] += tuple( l.item() if hasattr(l, 'item') else 0 for l in losses )
                n_sample += batch_size

                # stop after n_batch
                if n_batch is not None and k == n_batch - 1:
                    break
            detailed_loss[0][which][epoch_] /= n_sample

        scheduler.step()

        with torch.no_grad():
            for which in range(n_encoder):
                models[which].eval()
                instruments[which].eval()

                n_sample = 0
                for k, batch in enumerate(validloaders[which]):
                    batch_size = len(batch[0])
                    losses = get_losses(
                        models[which],
                        instruments[which],
                        batch,
                        aug_fct=aug_fcts[which],
                        similarity=similarity,
                        consistency=consistency,
                        loss_config=loss_configs[which],
                    )
                    # logging: validation
                    detailed_loss[1][which][epoch_] += tuple( l.item() if hasattr(l, 'item') else 0 for l in losses )
                    n_sample += batch_size

                    # stop after n_batch
                    if n_batch is not None and k == n_batch - 1:
                        break

                detailed_loss[1][which][epoch_] /= n_sample

        if verbose:
            mem_report()
            losses = tuple(detailed_loss[0, :, epoch_, :])
            vlosses = tuple(detailed_loss[1, :, epoch_, :])
            print('====> Epoch: %i'%(epoch))
            print('TRAINING Losses:', losses)
            print('VALIDATION Losses:', vlosses)

        if epoch_ % 5 == 0 or epoch_ == n_epoch - 1:
            args = models
            checkpoint(accelerator, args, optimizer, scheduler, n_encoder, outfile, detailed_loss)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("dir", help="data file directory")
    parser.add_argument("outfile", help="output file name")
    parser.add_argument("-n", "--latents", help="latent dimensionality", type=int, default=2)
    parser.add_argument("-b", "--batch_size", help="batch size", type=int, default=512)
    parser.add_argument("-l", "--batch_number", help="number of batches per epoch", type=int, default=None)
    parser.add_argument("-r", "--rate", help="learning rate", type=float, default=1e-3)
    parser.add_argument("-a", "--augmentation", help="add augmentation loss", action="store_true")
    parser.add_argument("-s", "--similarity", help="add similarity loss", action="store_true")
    parser.add_argument("-c", "--consistency", help="add consistency loss", action="store_true")
    parser.add_argument("-C", "--clobber", help="continue training of existing model", action="store_true")
    parser.add_argument("-v", "--verbose", help="verbose printing", action="store_true")
    args = parser.parse_args()

    # define instruments
    instruments = [ SDSS() ]
    n_encoder = len(instruments)

    # restframe wavelength for reconstructed spectra
    # Note: represents joint dataset wavelength range
    lmbda_min = 2359
    if n_encoder==1:lmbda_max = 9332
    else: lmbda_max = 10402
    bins = 7000
    wave_rest = torch.linspace(lmbda_min, lmbda_max, bins, dtype=torch.float32)
    if args.verbose:
        print ("Restframe:\t{:.0f} .. {:.0f} A ({} bins)".format(lmbda_min, lmbda_max, bins))

    # data loaders
    trainloaders = [ inst.get_data_loader(args.dir, which="train",  batch_size=args.batch_size, shuffle=True) for inst in instruments ]
    validloaders = [ inst.get_data_loader(args.dir, which="valid", batch_size=args.batch_size) for inst in instruments ]

    # get augmentation function
    if args.augmentation:
        aug_fcts = [ SDSS.augment_spectra, BOSS.augment_spectra ]
    else:
        aug_fcts = [ None,None ]

    # define training sequence
    SED = {"data":[True,False], "decoder":True}
    BED = {"data":[False,True], "decoder":True}
    SE = {"data":[True,False], "decoder":False}
    BE = {"data":[False,True], "decoder":False}
    D = {"data":[True,True],"encoder":[False,False],"decoder":True}
    SEBE = {"data":[True,True], "decoder":False}
    FULL = {"data":[True,True],"decoder":True}
    train_sequence = prepare_train([FULL])

    annealing_step = 0.10
    ANNEAL_SCHEDULE = np.arange(0,10,annealing_step)

    if args.verbose and args.similarity:
        print("similarity_slope:",len(ANNEAL_SCHEDULE),ANNEAL_SCHEDULE)

    # define and train the model
    n_hidden = (64, 256, 1024)
    models = [ SpectrumAutoencoder(instrument,
                                   wave_rest,
                                   n_latent=args.latents,
                                   n_hidden=n_hidden,
                                   n_aux=0
                                   )
              for instrument in instruments ]
    # use same decoder
    if n_encoder==2:models[1].decoder = models[0].decoder

    n_epoch = sum([item['iteration'] for item in train_sequence])
    init_t = time.time()
    if args.verbose:
        print("torch.cuda.device_count():",torch.cuda.device_count())
        print (f"--- Model {args.outfile} ---")

    # check if outfile already exists, continue only of -c is set
    if os.path.isfile(args.outfile) and not args.clobber:
        raise SystemExit("\nOutfile exists! Set option -C to continue training.")
    losses = None
    if os.path.isfile(args.outfile):
        if args.verbose:
            print (f"\nLoading file {args.outfile}")
        model, losses = load_model(args.outfile, models, instruments)

    # hyperparameters of the similarity and consistency losses, per instrument;
    # override entries here to tune training, e.g. loss_configs[0].restframe_mu = 6000
    loss_configs = [DEFAULT_LOSS_CONFIGS[instrument.__class__.__name__] for instrument in instruments]

    train(models, instruments, trainloaders, validloaders, n_epoch=n_epoch,
          n_batch=args.batch_number, lr=args.rate, aug_fcts=aug_fcts, similarity=args.similarity, consistency=args.consistency, outfile=args.outfile, losses=losses, verbose=args.verbose, loss_configs=loss_configs)

    if args.verbose:
        print("--- %s seconds ---" % (time.time()-init_t))
