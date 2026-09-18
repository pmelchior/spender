#!/usr/bin/env python

import argparse
import functools
import os
import time

import numpy as np
import torch
from torch import nn
from accelerate import Accelerator
from spender import SpectrumAutoencoder
from spender.data import desi
from spender.loss import LossConfig, get_losses
from spender.util import mem_report

# allows one to run fp16_train.py from home directory
import sys;sys.path.insert(1, './')

def prepare_train(seq,niter=800):
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

def get_all_parameters(model, instrument):
    model_params = list(model.encoder.parameters()) + list(model.decoder.parameters())
    dicts = [{'params':model_params}]

    n_parameters = sum(p.numel() for p in model_params if p.requires_grad)

    instr_params = list(instrument.parameters())
    if instr_params:
        dicts.append({'params':instr_params,'lr': 1e-4})
        n_parameters += sum(p.numel() for p in instr_params if p.requires_grad)
        print("parameter dict:",dicts[1])
    return dicts,n_parameters



def checkpoint(accelerator, model, outfile, losses):
    unwrapped = accelerator.unwrap_model(model).state_dict()

    accelerator.save({
        "model": unwrapped,
        "losses": losses,
    }, outfile)
    return

def load_model(filename, model, instrument):
    device = instrument.wave_obs.device
    model_struct = torch.load(filename, map_location=device)
    # backwards compat: encoder.mlp instead of encoder.mlp.mlp
    if 'encoder.mlp.mlp.0.weight' in model_struct['model'].keys():
        from collections import OrderedDict
        model_struct['model'] = OrderedDict([(k.replace('mlp.mlp', 'mlp'), v) for k, v in model_struct['model'].items()])
    # backwards compat: add instrument to encoder
    try:
        model.load_state_dict(model_struct['model'], strict=False)
    except RuntimeError:
        model_struct['model']['encoder.instrument.wave_obs'] = instrument.wave_obs
        model_struct['model']['encoder.instrument.skyline_mask'] = instrument.skyline_mask
        model.load_state_dict(model_struct['model'], strict=False)

    losses = model_struct['losses']
    return model, losses


def train(model,
          instrument,
          trainloader,
          validloader,
          n_epoch=200,
          outfile=None,
          losses=None,
          verbose=False,
          lr=1e-4,
          n_batch=50,
          aug_fct=None,
          similarity=True,
          consistency=True,
          loss_config=None,
          ):

    if loss_config is None:
        loss_config = LossConfig()

    model_parameters, n_parameters = get_all_parameters(model,instrument)

    if verbose:
        print("model parameters:", n_parameters)
        mem_report()

    ladder = build_ladder(train_sequence)
    optimizer = torch.optim.Adam(model_parameters, lr=lr, eps=1e-4)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, lr,
                                              total_steps=n_epoch)

    accelerator = Accelerator(mixed_precision='fp16')
    model = accelerator.prepare(model)
    instrument = accelerator.prepare(instrument)
    trainloader = accelerator.prepare(trainloader)
    validloader = accelerator.prepare(validloader)
    optimizer = accelerator.prepare(optimizer)

    # define losses to track: fidelity, similarity, consistency
    n_loss = 3
    epoch = 0
    if losses is None:
        detailed_loss = np.zeros((2, n_epoch, n_loss))
    else:
        try:
            epoch = len(losses[0])
            n_epoch += epoch
            detailed_loss = np.zeros((2, n_epoch, n_loss))
            detailed_loss[:, :epoch, :] = losses
            if verbose:
                print(f'====> Epoch: {epoch-1}')
                print('TRAINING Losses:', tuple(detailed_loss[0, epoch-1, :]))
                print('VALIDATION Losses:', tuple(detailed_loss[1, epoch-1, :]))
        except: # OK if losses are empty
            pass

    if outfile is None:
        outfile = "checkpoint.pt"

    for epoch_ in range(epoch, n_epoch):

        mode = train_sequence[ladder[epoch_ - epoch]]

        # turn on/off model decoder
        for p in model.decoder.parameters():
            p.requires_grad = mode['decoder']

        slope = ANNEAL_SCHEDULE[(epoch_ - epoch)%len(ANNEAL_SCHEDULE)]
        if n_epoch-epoch_<=10:
            slope=0 # turn off similarity
        loss_config.similarity_slope = slope

        if verbose and similarity:
            print("similarity info:",slope)

        # turn on/off encoder
        for p in model.encoder.parameters():
            p.requires_grad = mode['encoder']

        # optional: training on single dataset
        if mode['data']:
            model.train()
            instrument.train()

            n_sample = 0
            for k, batch in enumerate(trainloader):
                batch_size = len(batch[0])
                losses = get_losses(
                    model,
                    instrument,
                    batch,
                    aug_fct=aug_fct,
                    similarity=similarity,
                    consistency=consistency,
                    loss_config=loss_config,
                )
                # weighted combination of the individual losses for backprop
                fidelity_loss, sim_loss, cons_loss = losses
                loss = fidelity_loss + loss_config.similarity_amp * sim_loss + loss_config.consistency_amp * cons_loss
                accelerator.backward(loss)
                # clip gradients: stabilizes training with similarity
                accelerator.clip_grad_norm_(model_parameters[0]['params'], 1.0)
                # once per batch
                optimizer.step()
                optimizer.zero_grad()

                # logging: training
                detailed_loss[0][epoch_] += tuple( l.item() if hasattr(l, 'item') else 0 for l in losses )
                n_sample += batch_size

                # stop after n_batch
                if n_batch is not None and k == n_batch - 1:
                    break
            detailed_loss[0][epoch_] /= n_sample

        scheduler.step()

        with torch.no_grad():
            model.eval()
            instrument.eval()

            n_sample = 0
            for k, batch in enumerate(validloader):
                batch_size = len(batch[0])
                losses = get_losses(
                    model,
                    instrument,
                    batch,
                    aug_fct=aug_fct,
                    similarity=similarity,
                    consistency=consistency,
                    loss_config=loss_config,
                )
                # logging: validation
                detailed_loss[1][epoch_] += tuple( l.item() if hasattr(l, 'item') else 0 for l in losses )
                n_sample += batch_size

                # stop after n_batch
                if n_batch is not None and k == n_batch - 1:
                    break

            detailed_loss[1][epoch_] /= n_sample

        if verbose:
            mem_report()
            print('====> Epoch: %i'%(epoch))
            print('TRAINING Losses:', tuple(detailed_loss[0, epoch_, :]))
            print('VALIDATION Losses:', tuple(detailed_loss[1, epoch_, :]))

        if epoch_ % 5 == 0 or epoch_ == n_epoch - 1:
            checkpoint(accelerator, model, outfile, detailed_loss)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("dir", help="data file directory")
    parser.add_argument("outfile", help="output file name")
    parser.add_argument("-n", "--latents", help="latent dimensionality", type=int, default=2)
    parser.add_argument("-b", "--batch_size", help="batch size", type=int, default=512)
    parser.add_argument("-l", "--batch_number", help="number of batches per epoch", type=int, default=None)
    parser.add_argument("-r", "--rate", help="learning rate", type=float, default=1e-3)
    parser.add_argument("-zmax", "--z_max", help="constrain redshifts to z_max", type=float, default=0.8)
    parser.add_argument("-a", "--augmentation", help="add augmentation loss", action="store_true")
    parser.add_argument("-s", "--similarity", help="add similarity loss", action="store_true")
    parser.add_argument("-c", "--consistency", help="add consistency loss", action="store_true")
    parser.add_argument("-C", "--clobber", help="continue training of existing model", action="store_true")
    parser.add_argument("-v", "--verbose", help="verbose printing", action="store_true")
    args = parser.parse_args()

    # define instrument
    instrument = desi.DESI()

    # restframe wavelength for reconstructed spectra
    if args.z_max > 0.01:# DESI BGS
        lmbda_min = instrument.wave_obs[0]/(1.0+args.z_max) # 2000 A
        lmbda_max = instrument.wave_obs[-1] # 9824 A
        bins = 9780
    else: # DESI MWS
        lmbda_min = instrument.wave_obs[0]/(1.0+args.z_max)
        lmbda_max = instrument.wave_obs[-1]/(1.0-args.z_max)
        bins = int((lmbda_max-lmbda_min).item()/0.8)
    wave_rest = torch.linspace(lmbda_min, lmbda_max, bins, dtype=torch.float32)

    if args.verbose:
        print ("Restframe:\t{:.0f} .. {:.0f} A ({} bins)".format(lmbda_min, lmbda_max, bins))

    # data loaders
    trainloader = instrument.get_data_loader(args.dir, tag="Stars", which="train",  batch_size=args.batch_size, shuffle=True, shuffle_instance=True)
    validloader = instrument.get_data_loader(args.dir,  tag="Stars", which="valid", batch_size=args.batch_size, shuffle=True, shuffle_instance=True)

    # get augmentation function
    if args.augmentation:
        aug_fct = functools.partial(desi.DESI().augment_spectra, z_max=args.z_max)
    else:
        aug_fct = None

    # define training sequence
    FULL = {"data":True,"decoder":True}
    train_sequence = prepare_train([FULL])

    annealing_step = 0.1
    ANNEAL_SCHEDULE = np.arange(0.0,2.0,annealing_step)

    if args.verbose and args.similarity:
        print("similarity_slope:",len(ANNEAL_SCHEDULE),ANNEAL_SCHEDULE)

    # define and train the model
    n_hidden = (64, 256, 1024)
    model = SpectrumAutoencoder(instrument,
                                 wave_rest,
                                 n_latent=args.latents,
                                 n_hidden=n_hidden,
                                 act=[nn.LeakyReLU()]*(len(n_hidden)+1)
                                 )

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
        model, losses = load_model(args.outfile, model, instrument)
        non_zero = np.sum(losses[0],axis=1)>0
        losses = losses[:,non_zero,:]

    # hyperparameters of the similarity and consistency losses; override fields here
    # to tune training, e.g. LossConfig(restframe_mu=6000)
    loss_config = LossConfig()

    train(model, instrument, trainloader, validloader, n_epoch=n_epoch,
          n_batch=args.batch_number, lr=args.rate, aug_fct=aug_fct, similarity=args.similarity, consistency=args.consistency, outfile=args.outfile, losses=losses, verbose=args.verbose, loss_config=loss_config)

    if args.verbose:
        print("--- %s seconds ---" % (time.time()-init_t))
