#!/usr/bin/env python

import time, argparse
import numpy as np
import functools
import torch
from torch import nn
from torch import optim
from accelerate import Accelerator
from torchinterp1d import Interp1d

from batch_wrapper import get_data_loader, collect_batches, load_batch, save_batch
from instrument import get_instrument
from model import SpectrumAutoencoder
from util import load_model, permute_indices, mem_report, LogLinearDistribution, insert_jitters, jitter_redshift


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

def resample_to_restframe(wave_obs,wave_rest,y,w,z):
    wave_z = (wave_rest.unsqueeze(1)*(1 + z)).T
    wave_obs = wave_obs.repeat(y.shape[0],1)
    # resample restframe spectra on observed spectra
    yrest = Interp1d()(wave_obs,y,wave_z)
    wrest =  Interp1d()(wave_obs,w,wave_z)
    msk = (wave_z<=wave_obs.min())|(wave_z>=wave_obs.max())
    yrest[msk]=0
    wrest[msk]=0
    return yrest,wrest

def _similarity_loss(spec,w,s,individual=False,rand=[],wid=5,slope=0.5):
    batch_size, s_size = s.shape
    if rand==[]:rand = permute_indices(batch_size)
    new_w = 1.0/(w**(-1)+w[rand]**(-1))
    D = (new_w > 0).sum(dim=1)
    D[D<1]=1 # avoids NAN in loss function
    spec_sim = torch.sum(new_w*(spec[rand]-spec)**2,dim=1)/D
    s_sim = torch.sum((s[rand]-s)**2,dim=1)/s_size
    x = s_sim-spec_sim
    sim_loss = (torch.sigmoid(x)+torch.sigmoid(-slope*x-wid))*D
    if individual:
        return s_sim,spec_sim,sim_loss
    return sim_loss.sum()

def similarity_loss(instrument,model,spec,w,z,s,slope=0.5,individual=False):
    spec,w = resample_to_restframe(instrument.wave_obs,
                                   model.decoder.wave_rest,
                                   spec,w,z)
    return _similarity_loss(spec,w,s,slope=slope,individual=individual)

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
               augmentation=True,
               similarity=True,
               consistency=True,
               slope=0,
               mask_skyline=True,
               ):

    loss, sim_loss, s = _losses(model, instrument, batch, similarity=similarity, slope=slope, mask_skyline=mask_skyline)

    if augmentation:
        batch_copy = jitter_redshift(batch, mock_params, instrument)
        loss_, sim_loss_, s_ = _losses(model, instrument, batch_copy["batch"], similarity=similarity, slope=slope, mask_skyline=mask_skyline)
    else:
        loss_ = sim_loss_ = 0

    if augmentation and consistency:
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
          similarity=True,
          augmentation=True,
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
                    augmentation=augmentation,
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
                n_sample += (batch[1] > 0).sum().item()

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
                        augmentation=augmentation,
                        similarity=similarity,
                        consistency=consistency,
                        slope=slope,
                        mask_skyline=mask_skyline,
                    )
                    # logging: validation
                    detailed_loss[1][which][epoch] += tuple( l.item() if hasattr(l, 'item') else 0 for l in losses )
                    n_sample += (batch[1] > 0).sum().item()

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
          n_batch=args.batch_number, lr=args.rate, augmentation=args.augmentation, similarity=args.similarity, consistency=args.consistency, label=label, verbose=args.verbose)

    if args.verbose:
        print("--- %s seconds ---" % (time.time()-init_t))
