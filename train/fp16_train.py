#!/usr/bin/env python

import time, argparse
import numpy as np
import functools
import torch
from torch import nn
from torch import optim
from accelerate import Accelerator
from torchinterp1d import Interp1d

from spender import SpectrumAutoencoder, get_instrument, get_data_loader
from spender.util import mem_report, augment_spectra, resample_to_restframe


def prepare_train(seq,niter=500):
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
        #print("Adding %d parameters..."%sum(s))
    if instr_params != []:
        dicts.append({'params':instr_params,'lr': 1e-4})
        n_parameters += sum([p.numel() for p in instr_params if p.requires_grad])
        print("parameter dict:",dicts[1])
    return dicts,n_parameters

def consistency_loss(s, s_aug, individual=False):
    batch_size, s_size = s.shape
    x = torch.sum((s_aug - s)**2/1e-2,dim=1)/s_size
    sim_loss = torch.sigmoid(x)-0.5 # zero = perfect alignment
    if individual:
        return x, sim_loss
    return sim_loss.sum()

def similarity_loss(instrument, model, spec, w, z, s, slope=0.5, individual=False, wid=5):
    spec,w = resample_to_restframe(instrument.wave_obs,
                                   model.decoder.wave_rest,
                                   spec,w,z)

    batch_size, spec_size = spec.shape
    _, s_size = s.shape
    device = s.device

    # pairwise dissimilarity of spectra
    S = (spec[None,:,:] - spec[:,None,:])**2

    # pairwise weights
    non_zero = w > 0
    W = (1 / w)[None,:,:] + (1 / w)[:,None,:]
    W = (non_zero[None,:,:] * non_zero[:,None,:]) / W

    # dissimilarity of spectra
    # of order unity, larger for spectrum pairs with more comparable bins
    spec_sim = (W * S).sum(-1) / spec_size

    # dissimilarity of latents
    s_sim = ((s[None,:,:] - s[:,None,:])**2).sum(-1) / s_size

    # only give large loss of (dis)similarities are different (either way)
    x = s_sim-spec_sim
    sim_loss = torch.sigmoid(x)+torch.sigmoid(-slope*x-wid)

    if individual:
        return s_sim,spec_sim,sim_loss

    # total loss: sum over N^2 terms,
    # needs to have amplitude of N terms to compare to fidelity loss
    return sim_loss.sum() / batch_size

def _losses(model,
            instrument,
            batch,
            similarity=True,
            slope=0,
            mask_skyline=True,
           ):

    spec, w, z = batch
    if mask_skyline:
        w[:, instrument.skyline_mask] = 0

    # need the latents later on if similarity=True
    s = model.encode(spec, w=w, z=z)
    loss = model.loss(spec, w, instrument, z=z, s=s)

    if similarity:
        sim_loss = similarity_loss(instrument, model, spec, w, z, s, slope=slope)
    else: sim_loss = 0

    return loss, sim_loss, s

def get_losses(model,
               instrument,
               batch,
               aug_fct=None,
               similarity=True,
               consistency=True,
               slope=0,
               mask_skyline=True,
               ):

    loss, sim_loss, s = _losses(model, instrument, batch, similarity=similarity, slope=slope, mask_skyline=mask_skyline)

    if aug_fct is not None:
        batch_copy = aug_fct(batch, instrument)
        loss_, sim_loss_, s_ = _losses(model, instrument, batch_copy, similarity=similarity, slope=slope, mask_skyline=mask_skyline)
    else:
        loss_ = sim_loss_ = 0

    if consistency and aug_fct is not None:
        cons_loss = consistency_loss(s, s_)
    else:
        cons_loss = 0

    return loss, sim_loss, loss_, sim_loss_, cons_loss


def checkpoint(accelerator, args, optimizer, scheduler, n_encoder, label, losses):
    unwrapped = [accelerator.unwrap_model(args_i).state_dict() for args_i in args]

    model_unwrapped = unwrapped[:n_encoder]
    instruments_unwrapped = unwrapped[n_encoder:2*n_encoder]

    # checkpoints
    accelerator.save({
        "model": model_unwrapped,
        "instrument": instruments_unwrapped,
        "optimizer": optimizer.optimizer.state_dict(), # optimizer is an AcceleratedOptimizer object
        "scheduler": scheduler.state_dict(),
        "losses": losses,
    }, f'{label}.pt')
    return


def train(models,
          instruments,
          trainloaders,
          validloaders,
          n_epoch=200,
          label="",
          verbose=False,
          lr=1e-4,
          n_batch=50,
          mask_skyline=True,
          aug_fct=None,
          similarity=True,
          consistency=True,
          ):

    n_encoder = len(models)
    model_parameters, n_parameters = get_all_parameters(models,instruments)

    if verbose:
        print("model parameters:", n_parameters)
        mem_report()

    ladder = build_ladder(train_sequence)
    optimizer = optim.Adam(model_parameters, lr=lr, eps=1e-4)
    scheduler = optim.lr_scheduler.OneCycleLR(optimizer, lr,
                                              total_steps=n_epoch)

    accelerator = Accelerator(mixed_precision='fp16')
    models = [accelerator.prepare(model) for model in models]
    instruments = [accelerator.prepare(instrument) for instrument in instruments]
    trainloaders = [accelerator.prepare(loader) for loader in trainloaders]
    validloaders = [accelerator.prepare(loader) for loader in validloaders]
    optimizer = accelerator.prepare(optimizer)

    # define losses to track
    n_loss = 5
    detailed_loss = np.zeros((2, n_encoder, n_epoch, n_loss))

    for epoch in range(n_epoch):

        mode = train_sequence[ladder[epoch]]

        # turn on/off model decoder
        for p in models[0].decoder.parameters():
            p.requires_grad = mode['decoder']

        slope = ANNEAL_SCHEDULE[epoch%len(ANNEAL_SCHEDULE)]
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
                    aug_fct=aug_fct,
                    similarity=similarity,
                    consistency=consistency,
                    slope=slope,
                    mask_skyline=mask_skyline,
                )
                # sum up all losses
                loss = functools.reduce(lambda a, b: a+b , losses)
                accelerator.backward(loss)
                # clip gradients: stabilizes training with similarity
                accelerator.clip_grad_norm_(model_parameters[0]['params'], 1.0)
                # once per batch
                optimizer.step()
                optimizer.zero_grad()

                # logging: training
                detailed_loss[0][which][epoch] += tuple( l.item() if hasattr(l, 'item') else 0 for l in losses )
                n_sample += batch_size

                # stop after n_batch
                if n_batch is not None and k == n_batch - 1:
                    break
            detailed_loss[0][which][epoch] /= n_sample

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
                        aug_fct=aug_fct,
                        similarity=similarity,
                        consistency=consistency,
                        slope=slope,
                        mask_skyline=mask_skyline,
                    )
                    # logging: validation
                    detailed_loss[1][which][epoch] += tuple( l.item() if hasattr(l, 'item') else 0 for l in losses )
                    n_sample += batch_size

                    # stop after n_batch
                    if n_batch is not None and k == n_batch - 1:
                        break

                detailed_loss[1][which][epoch] /= n_sample

        if verbose:
            mem_report()
            losses = tuple(detailed_loss[0, :, epoch, :])
            vlosses = tuple(detailed_loss[1, :, epoch, :])
            print('====> Epoch: %i'%(epoch))
            print('TRAINING Losses:', losses)
            print('VALIDATION Losses:', vlosses)

        if epoch % 5 == 0 or epoch == n_epoch - 1:
            args = models + instruments
            checkpoint(accelerator, args, optimizer, scheduler, n_encoder, label, detailed_loss)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("dir", help="data file directory")
    parser.add_argument("label", help="output file label")
    parser.add_argument("-o", "--outdir", help="output file directory", default=".")
    parser.add_argument("-n", "--latents", help="latent dimensionality", type=int, default=2)
    parser.add_argument("-b", "--batch_size", help="batch size", type=int, default=512)
    parser.add_argument("-l", "--batch_number", help="number of batches per epoch", type=int, default=None)
    parser.add_argument("-r", "--rate", help="learning rate", type=float, default=1e-3)
    parser.add_argument("-a", "--augmentation", help="add augmentation loss", action="store_true")
    parser.add_argument("-s", "--similarity", help="add similarity loss", action="store_true")
    parser.add_argument("-c", "--consistency", help="add consistency loss", action="store_true")
    parser.add_argument("-v", "--verbose", help="verbose printing", action="store_true")
    args = parser.parse_args()

    # define instruments
    instrument_names = ["SDSS", "BOSS"]
    instruments = [ get_instrument(name) for name in instrument_names ]
    n_encoder = len(instrument_names)

    # restframe wavelength for reconstructed spectra
    # Note: represents joint dataset wavelength range
    lmbda_min = 2359
    lmbda_max = 10402
    bins = 7000
    wave_rest = torch.linspace(lmbda_min, lmbda_max, bins, dtype=torch.float32)
    if args.verbose:
        print ("Restframe:\t{:.0f} .. {:.0f} A ({} bins)".format(lmbda_min, lmbda_max, bins))

    # data loaders
    trainloaders = [ get_data_loader(args.dir, name, which="train",  batch_size=args.batch_size, shuffle=True) for name in instrument_names ]
    validloaders = [ get_data_loader(args.dir, name, which="valid", batch_size=args.batch_size) for name in instrument_names ]

    # get augmentation function
    if args.augmentation:
        aug_fct = augment_spectra
    else:
        aug_fct = None

    # define training sequence
    SED = {"data":[True,False], "decoder":True}
    BED = {"data":[False,True], "decoder":True}
    SE = {"data":[True,False], "decoder":False}
    BE = {"data":[False,True], "decoder":False}
    D = {"data":[True,True],"encoder":[False,False],"decoder":True}
    SEBE = {"data":[True,True], "decoder":False}
    FULL = {"data":[True,True],"decoder":True}
    train_sequence = prepare_train([FULL])

    annealing_step = 0.05
    ANNEAL_SCHEDULE = np.arange(0.2,1,annealing_step)
    if args.verbose and args.similarity:
        print("similarity_slope:",len(ANNEAL_SCHEDULE),ANNEAL_SCHEDULE)

    label = "%s/dataonly-%s.%d" % (args.outdir, args.label, args.latents)

    uniform_njit = [100,300]
    mock_params = [[0.4,uniform_njit]]#,[0.1,uniform_njit]]
    ncopy = len(mock_params)

    # define and train the model
    n_hidden = (64, 256, 1024)
    models = [ SpectrumAutoencoder(wave_rest,
                                    n_latent=args.latents,
                                    n_hidden=n_hidden,
                                    normalize=True),
              ] * 2
    # use same decoder
    models[1].decoder = models[0].decoder

    n_epoch = sum([item['iteration'] for item in train_sequence])
    init_t = time.time()
    if args.verbose:
        print("torch.cuda.device_count():",torch.cuda.device_count())
        print ("--- Model %s ---" % label)

    train(models, instruments, trainloaders, validloaders, n_epoch=n_epoch,
          n_batch=args.batch_number, lr=args.rate, aug_fct=aug_fct, similarity=args.similarity, consistency=args.consistency, label=label, verbose=args.verbose)

    if args.verbose:
        print("--- %s seconds ---" % (time.time()-init_t))
