import torch
import csv
import logging
import os
import pickle
import time

from argparse import Namespace
from logging import Logger
from typing import List
from torch.utils.data import DataLoader
from torch import distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from grover.data import MolCollator
from grover.data import StandardScaler
from grover.util.metrics import get_metric_func
from grover.util.nn_utils import initialize_weights, param_count
from grover.util.scheduler import NoamLR
from grover.util.utils import build_optimizer, build_lr_scheduler, makedirs, load_checkpoint, get_loss_func, \
    save_checkpoint, build_model
from grover.util.utils import get_class_sizes, get_data, split_data, get_task_names

import copy

def forward_backward_step(model, micro_batches, args, forward_only, loss_func):

    loss_func = get_loss_func(args, model)
    #disable_grad_sync()
    num_micro_batch = args.num_micro_batch
    model_parallel_size = args.model_parallel_size
    num_warmup_microbatches = min(model_parallel_size - args.node_rank -1, num_micro_batch)
    num_microbatches_remaining = num_micro_batch - num_warmup_microbatches

    f_atoms_size = [micro_batches[i]["graph_input"][0].size(0) for i in range(len(micro_batches))]
    f_bonds_size = [micro_batches[i]["graph_input"][1].size(0) for i in range(len(micro_batches))]

    target_lst = [micro_batches[i]["targets"] for i in range(len(micro_batches))]

    world_rank = dist.get_rank()

    input_batches, output_batches = [], []
    if not forward_only:
        input_batches, output_batches = [], []
    losses = []
    atom_output_lst, bond_output_lst = [], []

    # Warmup forward passes
    # print(f"num_warmup_microbatches: {num_warmup_microbatches}")
    for i in range(num_warmup_microbatches):
        micro_batch = micro_batches[i]
        batch = micro_batch["graph_input"]
        #targets = micro_batch["targets"]
        #target_lst.append(targets)
        # targets["av_task"] = targets["av_task"].cuda()
        # targets["bv_task"] = targets["bv_task"].cuda()
        # targets["fg_task"] = targets["fg_task"].cuda()
        f_atoms_original, f_bonds_original, a2b, b2a, b2revb, a_scope, b_scope, a2a = batch
        # Recv forward
        if args.node_rank != 0:
            f_atoms_feature = torch.empty(f_atoms_size[i], args.hidden_size).cuda()
            f_bonds_feature = torch.empty(f_bonds_size[i], args.hidden_size).cuda()
            # print(f"recv fw: {f_atoms_size[i]} at line 1974 to {world_rank-args.data_parallel_size}, micro_batch: {i}")
            dist.recv(tensor=f_atoms_feature.cuda(), src=world_rank-args.data_parallel_size)
            dist.recv(tensor=f_bonds_feature.cuda(), src=world_rank-args.data_parallel_size)
            
            f_atoms_feature.requires_grad = True
            f_bonds_feature.requires_grad = True
        else:
            f_atoms_feature = None
            f_bonds_feature = None
        input_batch = (f_atoms_original, f_bonds_original, a2b, b2a, b2revb, a_scope, b_scope, 
                        a2a, f_atoms_feature, f_bonds_feature)
        
        # Forward Step
        output_batch = model(input_batch)
        if args.node_rank!=args.model_parallel_size - 1:
            atom_output, bond_output = output_batch
        else:
            atom_output, bond_output, _, _, _, _, _, _, _, _ = output_batch

        atom_output_lst.append(atom_output)
        bond_output_lst.append(bond_output)
        # Send FW
        # print(f"args.node_rank: {args.node_rank}, args.model_parallel_size: {args.model_parallel_size}")
        if args.node_rank != args.model_parallel_size - 1:
            # print(f"send fw: {atom_output.size()} at line 1993 to {world_rank+args.data_parallel_size}, micro_batch: {i}")
            dist.send(tensor=atom_output.cuda(), dst=world_rank + args.data_parallel_size)
            dist.send(tensor=bond_output.cuda(), dst=world_rank + args.data_parallel_size)
        
        if not forward_only:
            input_batches.append(input_batch)
            output_batches.append(output_batch)

    if num_microbatches_remaining > 0:
        # recv forward
        if args.node_rank != 0:
            f_atoms_feature = torch.empty(f_atoms_size[num_warmup_microbatches], args.hidden_size).cuda()
            f_bonds_feature = torch.empty(f_bonds_size[num_warmup_microbatches], args.hidden_size).cuda()
            # print(f"recv fw: {f_atoms_feature.size()} at line 2014 from {world_rank-args.data_parallel_size}, micro_batch: {num_warmup_microbatches}")
            dist.recv(tensor=f_atoms_feature, src=world_rank - args.data_parallel_size)
            dist.recv(tensor=f_bonds_feature, src=world_rank - args.data_parallel_size)
            
            f_atoms_feature.requires_grad = True
            f_bonds_feature.requires_grad = True
        else:
            f_atoms_feature = None
            f_bonds_feature = None
        micro_batch = micro_batches[num_warmup_microbatches]
        # targets = micro_batch["targets"]
        micro_batch = micro_batch["graph_input"]
        input_batch = (micro_batch[0], micro_batch[1], micro_batch[2], micro_batch[3], micro_batch[4], micro_batch[5], micro_batch[6], micro_batch[7], f_atoms_feature, f_bonds_feature)
        input_batches.append(input_batch)
        # if args.node_rank == 3:
        #     targets["av_task"] = targets["av_task"].cuda()
        #     targets["bv_task"] = targets["bv_task"].cuda()
        #     targets["fg_task"] = targets["fg_task"].cuda()
        #     target_lst.append(targets)

    # 1F1B
    # print("\nStart 1F1B")
    for i in range(num_microbatches_remaining):

        last_iteration = i == (num_microbatches_remaining - 1)

        # Forward Step
        output_batch = model(input_batch)

        if args.node_rank==args.model_parallel_size - 1:
            loss = output_batch
        else:
            atom_output, bond_output = output_batch
            atom_output_lst.append(atom_output)
            bond_output_lst.append(bond_output)

        if forward_only:
            # Send FW
            if args.node_rank != args.model_parallel_size - 1:
                # print(f"send fw: {atom_output.size()} at line 2047 to {world_rank+args.data_parallel_size}")
                dist.send(tensor=atom_output.cuda(), dst=world_rank+args.data_parallel_size)
                dist.send(tensor=bond_output.cuda(), dst=world_rank+args.data_parallel_size)
            else:
                target = target_lst[0]
                target["av_task"] = target["av_task"].cuda()
                target["bv_task"] = target["bv_task"].cuda()
                target["fg_task"] = target["fg_task"].cuda()
                loss = loss_func(loss, target)
                target_lst = target_lst[1:]
                losses.append(loss)
            if not last_iteration:
                # Recv forward
                if args.node_rank != 0:
                    # print(f"recv fw: {f_atoms_feature.size()} at line 2052 from {world_rank-args.data_parallel_size}")
                    f_atoms_feature = torch.empty(f_atoms_size[i + num_warmup_microbatches + 1], args.hidden_size).cuda()
                    f_bonds_feature = torch.empty(f_bonds_size[i + num_warmup_microbatches + 1], args.hidden_size).cuda()
                    dist.recv(tensor=f_atoms_feature.cuda(), src=world_rank-args.data_parallel_size)
                    dist.recv(tensor=f_bonds_feature.cuda(), src=world_rank-args.data_parallel_size)
                micro_batch = micro_batches[i + num_warmup_microbatches + 1]
                targets = micro_batch["targets"]
                targets["av_task"] = targets["av_task"].cuda()
                targets["bv_task"] = targets["bv_task"].cuda()
                targets["fg_task"] = targets["fg_task"].cuda()
                # target_lst.append(targets)
                micro_batch = micro_batch["graph_input"]
                input_batch = (micro_batch[0], micro_batch[1], micro_batch[2], micro_batch[3], micro_batch[4], micro_batch[5], micro_batch[6], micro_batch[7], f_atoms_feature, f_bonds_feature)
        else:
            # Send FW recv BW
            if args.node_rank != args.model_parallel_size - 1:

                atom_output_grad, bond_output_grad = torch.empty((f_atoms_size[i], args.hidden_size)), torch.empty((f_bonds_size[i], args.hidden_size))
                # Recv BW
                # print(f"recv bw: {atom_output_grad.size()} at line 2054 from {world_rank + args.data_parallel_size}, micro_batch: {len(micro_batches) - i}")
                dist.recv(tensor=atom_output_grad.cuda(), src=world_rank + args.data_parallel_size)
                dist.recv(tensor=bond_output_grad.cuda(), src=world_rank + args.data_parallel_size)

                # Send FW
                # print(f"send fw: {atom_output.size()} at line 2052 to {world_rank + args.data_parallel_size}, micro_batch: {i + num_warmup_microbatches}")
                dist.send(tensor=atom_output.cuda(), dst=world_rank + args.data_parallel_size)
                dist.send(tensor=bond_output.cuda(), dst=world_rank + args.data_parallel_size)

                if f_atoms_feature is not None:
                    f_atoms_feature.retain_grad()
                    f_bonds_feature.retain_grad()

                # Backward Step
                atom_output_lst[0].backward(atom_output_grad.cuda(), retain_graph=True)
                bond_output_lst[0].backward(bond_output_grad.cuda(), retain_graph=True)
                atom_output_lst = atom_output_lst[1:]
                bond_output_lst = bond_output_lst[1:]
            else: # if last rank
                target = target_lst[0]
                target["av_task"] = target["av_task"].cuda()
                target["bv_task"] = target["bv_task"].cuda()
                target["fg_task"] = target["fg_task"].cuda()
                loss = loss_func(loss, target)
                target_lst = target_lst[1:]
                losses.append(loss)
                loss, _, _, _, _, _, _ = loss
                loss.backward(retain_graph=True)

            if last_iteration:
                if args.node_rank != 0:
                    # send backward
                    # print(f"send bw: {input_batches[0][-2].grad.size()} at line 2080 to {world_rank - args.data_parallel_size}")
                    dist.send(tensor=input_batches[0][-2].grad, dst=world_rank - args.data_parallel_size)
                    dist.send(tensor=input_batches[0][-1].grad, dst=world_rank - args.data_parallel_size)
                    input_batches = input_batches[1:]
                input_batch = None
                    
            else:
                # Send backward
                if args.node_rank != 0:
                    # print(f"send bw: {input_batches[0][-2].grad.size()} at line 2085 to {world_rank - args.data_parallel_size}, micro_batch: {i + num_warmup_microbatches}")
                    dist.send(tensor=input_batches[0][-2].grad, dst=world_rank - args.data_parallel_size)
                    dist.send(tensor=input_batches[0][-1].grad, dst=world_rank - args.data_parallel_size)
                    input_batches = input_batches[1:]
                    
                # Recv forward
                if args.node_rank != 0:
                    f_atoms_feature = torch.empty(f_atoms_size[i + num_warmup_microbatches + 1], args.hidden_size).cuda()
                    f_bonds_feature = torch.empty(f_bonds_size[i + num_warmup_microbatches + 1], args.hidden_size).cuda()
                    # print(f"recv fw: {f_atoms_feature.size()} at line 2089 from {world_rank - args.data_parallel_size}, micro_batch: {i + num_warmup_microbatches}")
                    dist.recv(tensor=f_atoms_feature.cuda(), src=world_rank - args.data_parallel_size)
                    dist.recv(tensor=f_bonds_feature.cuda(), src=world_rank - args.data_parallel_size)
                    f_atoms_feature.requires_grad = True
                    f_bonds_feature.requires_grad = True
                else:
                    f_atoms_feature = None
                    f_bonds_feature = None
                micro_batch = micro_batches[i + num_warmup_microbatches + 1]
                # targets = micro_batch["targets"]
                micro_batch = micro_batch["graph_input"]
                input_batch = (micro_batch[0], micro_batch[1], micro_batch[2], micro_batch[3], micro_batch[4], micro_batch[5], micro_batch[6], micro_batch[7], f_atoms_feature, f_bonds_feature)
                input_batches.append(input_batch)
                # if args.node_rank == 3:
                #     targets["av_task"] = targets["av_task"].cuda()
                #     targets["bv_task"] = targets["bv_task"].cuda()
                #     targets["fg_task"] = targets["fg_task"].cuda()
                #     target_lst.append(targets)
                    
    
    # cooldown phase
    # print("\nStart cooldown phase")
    if not forward_only:
        for i in range(num_warmup_microbatches):

            if i == num_warmup_microbatches - 1:
                if args.node_rank == 0:
                    #enable_grad_sync()
                    pass
            atom_output = atom_output_lst[0]
            bond_output = bond_output_lst[0]
            atom_output_grad, bond_output_grad = torch.zeros_like(atom_output), torch.zeros_like(bond_output)
            
            if args.node_rank != args.model_parallel_size - 1:
                # recv backward
                # print(f"recv bw: {atom_output_grad.size()} at line 2113 from {world_rank+args.data_parallel_size}, micro_batch: {i}")
                dist.recv(tensor=atom_output_grad.cuda(), src=world_rank+args.data_parallel_size)
                dist.recv(tensor=bond_output_grad.cuda(), src=world_rank+args.data_parallel_size)
                
                # Backward Step
                atom_output.backward([atom_output_grad], retain_graph=True)
                bond_output.backward([bond_output_grad], retain_graph=True)
                atom_output_lst = atom_output_lst[1:]
                bond_output_lst = bond_output_lst[1:]
            else:
                loss = output_batch
                target = target_lst[0]
                target["av_task"] = target["av_task"].cuda()
                target["bv_task"] = target["bv_task"].cuda()
                target["fg_task"] = target["fg_task"].cuda()
                loss = loss_func(loss, target)
                target_lst = target_lst[1:]
                losses.append(loss)
                loss, _, _, _, _, _, _ = loss
                loss.backward(retain_graph=True)
            if args.node_rank != 0:
                # send backward
                # print(f"send bw: {input_batches[0][-2].grad.size()} at line 2127 to {world_rank-args.data_parallel_size}, micro_batch: {i}")
                dist.send(tensor=input_batches[0][-2].grad, dst=world_rank-args.data_parallel_size)
                dist.send(tensor=input_batches[0][-1].grad, dst=world_rank-args.data_parallel_size)
                # release retained graph
                input_batches = input_batches[1:]
            
    torch.cuda.empty_cache()

    # calculate loss
    if args.node_rank==3:
        if not forward_only:
            # preds = torch.stack(preds)
            # loss = loss_func(preds, targets) * class_weights * mask
            # loss = loss.sum() / mask.sum()
            pass
    else:
        loss = None
                
    return losses


def forward_backward_step_for_max_pipeline(model, micro_batches, args, forward_only, loss_func):

    if args.node_rank<4:
        loss_func = get_loss_func(args, model)
        #disable_grad_sync()
        num_micro_batch = args.num_micro_batch
        model_parallel_size = args.model_parallel_size//2
        num_warmup_microbatches = min(model_parallel_size - args.node_rank-1, num_micro_batch)
        num_microbatches_remaining = num_micro_batch - num_warmup_microbatches

        f_atoms_size = [micro_batches[i]["graph_input"][0].size(0) for i in range(len(micro_batches))]

        target_lst = [micro_batches[i]["targets"] for i in range(len(micro_batches))]

        world_rank = dist.get_rank()

        input_batches, output_batches = [], []
        if not forward_only:
            input_batches, output_batches = [], []
        losses = []
        atom_output_lst, bond_output_lst = [], []

        # Warmup forward passes
        for i in range(num_warmup_microbatches):
            micro_batch = micro_batches[i]
            batch = micro_batch["graph_input"]
            f_atoms_original, f_bonds_original, a2b, b2a, b2revb, a_scope, b_scope, a2a = batch
            # Recv forward
            if args.node_rank != 0:
                f_atoms_feature = torch.empty(f_atoms_size[i], args.hidden_size).cuda()
                f_bonds_feature = None
                # print(f"recv fw: {f_atoms_size[i]} at line 1974 to {world_rank-args.data_parallel_size}, micro_batch: {i}")
                dist.recv(tensor=f_atoms_feature.cuda(), src=world_rank-args.data_parallel_size)
            else:
                f_atoms_feature = None
                f_bonds_feature = None
            input_batch = (f_atoms_original, f_bonds_original, a2b, b2a, b2revb, a_scope, b_scope, 
                            a2a, f_atoms_feature, f_bonds_feature)
            
            # Forward Step
            output_batch = model(input_batch)
            if args.node_rank!=3:
                atom_output, _ = output_batch
            else:
                atom_output, _, _, _, _, _, _, _, _, _ = output_batch

            atom_output_lst.append(atom_output)
            # Send FW
            # print(f"args.node_rank: {args.node_rank}, args.model_parallel_size: {args.model_parallel_size}")
            if args.node_rank != 3:
                # print(f"send fw: {atom_output.size()} at line 1993 to {world_rank+args.data_parallel_size}, micro_batch: {i}")
                dist.send(tensor=atom_output.cuda(), dst=world_rank + args.data_parallel_size)
            
            if not forward_only:
                input_batches.append(input_batch)
                output_batches.append(output_batch)

        if num_microbatches_remaining > 0:
            # recv forward
            if args.node_rank != 0:
                f_atoms_feature = torch.empty(f_atoms_size[num_warmup_microbatches], args.hidden_size).cuda()
                f_bonds_feature = None
                # print(f"recv fw: {f_atoms_feature.size()} at line 2014 from {world_rank-args.data_parallel_size}, micro_batch: {num_warmup_microbatches}")
                dist.recv(tensor=f_atoms_feature, src=world_rank - args.data_parallel_size)
                
                f_atoms_feature.requires_grad = True
            else:
                f_atoms_feature = None
                f_bonds_feature = None
            micro_batch = micro_batches[num_warmup_microbatches]
            # targets = micro_batch["targets"]
            micro_batch = micro_batch["graph_input"]
            input_batch = (micro_batch[0], micro_batch[1], micro_batch[2], micro_batch[3], micro_batch[4], micro_batch[5], micro_batch[6], micro_batch[7], f_atoms_feature, f_bonds_feature)
            input_batches.append(input_batch)
            # if args.node_rank == 3:
            #     targets["av_task"] = targets["av_task"].cuda()
            #     targets["bv_task"] = targets["bv_task"].cuda()
            #     targets["fg_task"] = targets["fg_task"].cuda()
            #     target_lst.append(targets)

        # 1F1B
        # print("\nStart 1F1B")
        for i in range(num_microbatches_remaining):

            last_iteration = i == (num_microbatches_remaining - 1)

            # Forward Step
            output_batch = model(input_batch)

            if args.node_rank==3:
                loss = output_batch
            else:
                atom_output, _ = output_batch
                atom_output_lst.append(atom_output)

            if forward_only:
                # Send FW
                if args.node_rank ==3:
                    # print(f"send fw: {atom_output.size()} at line 2047 to {world_rank+args.data_parallel_size}")
                    dist.send(tensor=atom_output.cuda(), dst=world_rank+args.data_parallel_size)
                else:
                    target = target_lst[0]
                    target["av_task"] = target["av_task"].cuda()
                    target["bv_task"] = target["bv_task"].cuda()
                    target["fg_task"] = target["fg_task"].cuda()
                    loss = loss_func(loss, target)
                    target_lst = target_lst[1:]
                    losses.append(loss)
                if not last_iteration:
                    # Recv forward
                    if args.node_rank != 0:
                        # print(f"recv fw: {f_atoms_feature.size()} at line 2052 from {world_rank-args.data_parallel_size}")
                        f_atoms_feature = torch.empty(f_atoms_size[i + num_warmup_microbatches + 1], args.hidden_size).cuda()
                        f_bonds_feature = None
                        dist.recv(tensor=f_atoms_feature.cuda(), src=world_rank-args.data_parallel_size)
                    micro_batch = micro_batches[i + num_warmup_microbatches + 1]
                    targets = micro_batch["targets"]
                    targets["av_task"] = targets["av_task"].cuda()
                    targets["bv_task"] = targets["bv_task"].cuda()
                    targets["fg_task"] = targets["fg_task"].cuda()
                    # target_lst.append(targets)
                    micro_batch = micro_batch["graph_input"]
                    input_batch = (micro_batch[0], micro_batch[1], micro_batch[2], micro_batch[3], micro_batch[4], micro_batch[5], micro_batch[6], micro_batch[7], f_atoms_feature, f_bonds_feature)
            else:
                # Send FW recv BW
                if args.node_rank != 3:

                    atom_output_grad = torch.empty((f_atoms_size[i], args.hidden_size))
                    # Recv BW
                    # print(f"recv bw: {atom_output_grad.size()} at line 2054 from {world_rank + args.data_parallel_size}, micro_batch: {len(micro_batches) - i}")
                    dist.recv(tensor=atom_output_grad.cuda(), src=world_rank + args.data_parallel_size)

                    # Send FW
                    # print(f"send fw: {atom_output.size()} at line 2052 to {world_rank + args.data_parallel_size}, micro_batch: {i + num_warmup_microbatches}")
                    dist.send(tensor=atom_output.cuda(), dst=world_rank + args.data_parallel_size)

                    if f_atoms_feature is not None:
                        f_atoms_feature.retain_grad()

                    # Backward Step
                    atom_output_lst[0].backward(atom_output_grad.cuda(), retain_graph=True)
                    atom_output_lst = atom_output_lst[1:]
                else: # if last rank
                    target = target_lst[0]
                    target["av_task"] = target["av_task"].cuda()
                    target["bv_task"] = target["bv_task"].cuda()
                    target["fg_task"] = target["fg_task"].cuda()
                    loss = loss_func(loss, target)
                    target_lst = target_lst[1:]
                    losses.append(loss)
                    loss, _, _, _, _, _, _ = loss
                    loss.backward(retain_graph=True)

                if last_iteration:
                    if args.node_rank != 0:
                        # send backward
                        # print(f"send bw: {input_batches[0][-2].grad.size()} at line 2080 to {world_rank - args.data_parallel_size}")
                        dist.send(tensor=input_batches[0][-2].grad, dst=world_rank - args.data_parallel_size)
                        input_batches = input_batches[1:]
                    input_batch = None
                        
                else:
                    # Send backward
                    if args.node_rank != 0:
                        # print(f"send bw: {input_batches[0][-2].grad.size()} at line 2085 to {world_rank - args.data_parallel_size}, micro_batch: {i + num_warmup_microbatches}")
                        dist.send(tensor=input_batches[0][-2].grad, dst=world_rank - args.data_parallel_size)
                        dist.send(tensor=input_batches[0][-1].grad, dst=world_rank - args.data_parallel_size)
                        input_batches = input_batches[1:]
                        
                    # Recv forward
                    if args.node_rank != 0:
                        f_atoms_feature = torch.empty(f_atoms_size[i + num_warmup_microbatches + 1], args.hidden_size).cuda()
                        f_bonds_feature = None
                        # print(f"recv fw: {f_atoms_feature.size()} at line 2089 from {world_rank - args.data_parallel_size}, micro_batch: {i + num_warmup_microbatches}")
                        dist.recv(tensor=f_atoms_feature.cuda(), src=world_rank - args.data_parallel_size)
                        f_atoms_feature.requires_grad = True
                    else:
                        f_atoms_feature = None
                        f_bonds_feature = None
                    micro_batch = micro_batches[i + num_warmup_microbatches + 1]
                    micro_batch = micro_batch["graph_input"]
                    input_batch = (micro_batch[0], micro_batch[1], micro_batch[2], micro_batch[3], micro_batch[4], micro_batch[5], micro_batch[6], micro_batch[7], f_atoms_feature, f_bonds_feature)
                    input_batches.append(input_batch)
                        
        
        # cooldown phase
        # print("\nStart cooldown phase")
        if not forward_only:
            for i in range(num_warmup_microbatches):
                if i == num_warmup_microbatches - 1:
                    if args.node_rank == 0:
                        #enable_grad_sync()
                        pass
                atom_output = atom_output_lst[0]
                atom_output_grad = torch.zeros_like(atom_output)
                
                if args.node_rank != 3:
                    # recv backward
                    # print(f"recv bw: {atom_output_grad.size()} at line 2113 from {world_rank+args.data_parallel_size}, micro_batch: {i}")
                    dist.recv(tensor=atom_output_grad.cuda(), src=world_rank+args.data_parallel_size)
                    
                    # Backward Step
                    atom_output.backward([atom_output_grad], retain_graph=True)
                    bond_output.backward([bond_output_grad], retain_graph=True)
                    atom_output_lst = atom_output_lst[1:]
                    bond_output_lst = bond_output_lst[1:]
                else:
                    loss = output_batch
                    target = target_lst[0]
                    target["av_task"] = target["av_task"].cuda()
                    target["bv_task"] = target["bv_task"].cuda()
                    target["fg_task"] = target["fg_task"].cuda()
                    loss = loss_func(loss, target)
                    target_lst = target_lst[1:]
                    losses.append(loss)
                    loss, _, _, _, _, _, _ = loss
                    loss.backward(retain_graph=True)
                if args.node_rank != 0:
                    # send backward
                    # print(f"send bw: {input_batches[0][-2].grad.size()} at line 2127 to {world_rank-args.data_parallel_size}, micro_batch: {i}")
                    dist.send(tensor=input_batches[0][-2].grad, dst=world_rank-args.data_parallel_size)
                    # release retained graph
                    input_batches = input_batches[1:]
                
        torch.cuda.empty_cache()

        # calculate loss
        if args.node_rank==3:
            if not forward_only:
                pass
        else:
            loss = None
                    
        return losses

    else:
        loss_func = get_loss_func(args, model)
        #disable_grad_sync()
        num_micro_batch = args.num_micro_batch
        model_parallel_size = args.model_parallel_size//2
        num_warmup_microbatches = min(model_parallel_size - (args.node_rank-4)-1, num_micro_batch)
        num_microbatches_remaining = num_micro_batch - num_warmup_microbatches

        f_bonds_size = [micro_batches[i]["graph_input"][1].size(0) for i in range(len(micro_batches))]

        target_lst = [micro_batches[i]["targets"] for i in range(len(micro_batches))]

        world_rank = dist.get_rank()

        input_batches, output_batches = [], []
        if not forward_only:
            input_batches, output_batches = [], []
        losses = []
        bond_output_lst = []

        # Warmup forward passes
        for i in range(num_warmup_microbatches):
            micro_batch = micro_batches[i]
            batch = micro_batch["graph_input"]
            f_atoms_original, f_bonds_original, a2b, b2a, b2revb, a_scope, b_scope, a2a = batch
            # Recv forward
            if args.node_rank != 4:
                f_atoms_feature = None
                f_bonds_feature = torch.empty(f_bonds_size[i], args.hidden_size).cuda()
                # print(f"recv fw: {f_atoms_size[i]} at line 1974 to {world_rank-args.data_parallel_size}, micro_batch: {i}")
                dist.recv(tensor=f_bonds_feature.cuda(), src=world_rank-args.data_parallel_size)
                
                f_bonds_feature.requires_grad = True
            else:
                f_atoms_feature = None
                f_bonds_feature = None
            input_batch = (f_atoms_original, f_bonds_original, a2b, b2a, b2revb, a_scope, b_scope, 
                            a2a, f_atoms_feature, f_bonds_feature)
            
            # Forward Step
            output_batch = model(input_batch)
            if args.node_rank!=7:
                _, bond_output = output_batch
            else:
                _, bond_output, _, _, _, _, _, _, _, _ = output_batch

            bond_output_lst.append(bond_output)
            # Send FW
            # print(f"args.node_rank: {args.node_rank}, args.model_parallel_size: {args.model_parallel_size}")
            if args.node_rank != args.model_parallel_size - 1:
                # print(f"send fw: {atom_output.size()} at line 1993 to {world_rank+args.data_parallel_size}, micro_batch: {i}")
                dist.send(tensor=bond_output.cuda(), dst=world_rank + args.data_parallel_size)
            
            if not forward_only:
                input_batches.append(input_batch)
                output_batches.append(output_batch)

        if num_microbatches_remaining > 0:
            # recv forward
            if args.node_rank != 4:
                f_atoms_feature = None
                f_bonds_feature = torch.empty(f_bonds_size[num_warmup_microbatches], args.hidden_size).cuda()
                # print(f"recv fw: {f_atoms_feature.size()} at line 2014 from {world_rank-args.data_parallel_size}, micro_batch: {num_warmup_microbatches}")
                dist.recv(tensor=f_bonds_feature, src=world_rank - args.data_parallel_size)
                f_bonds_feature.requires_grad = True
            else:
                f_atoms_feature = None
                f_bonds_feature = None
            micro_batch = micro_batches[num_warmup_microbatches]
            # targets = micro_batch["targets"]
            micro_batch = micro_batch["graph_input"]
            input_batch = (micro_batch[0], micro_batch[1], micro_batch[2], micro_batch[3], micro_batch[4], micro_batch[5], micro_batch[6], micro_batch[7], f_atoms_feature, f_bonds_feature)
            input_batches.append(input_batch)
            # if args.node_rank == 3:
            #     targets["av_task"] = targets["av_task"].cuda()
            #     targets["bv_task"] = targets["bv_task"].cuda()
            #     targets["fg_task"] = targets["fg_task"].cuda()
            #     target_lst.append(targets)

        # 1F1B
        # print("\nStart 1F1B")
        for i in range(num_microbatches_remaining):

            last_iteration = i == (num_microbatches_remaining - 1)

            # Forward Step
            output_batch = model(input_batch)

            if args.node_rank==7:
                loss = output_batch
            else:
                _, bond_output = output_batch
                bond_output_lst.append(bond_output)

            if forward_only:
                # Send FW
                if args.node_rank != args.model_parallel_size - 1:
                    # print(f"send fw: {atom_output.size()} at line 2047 to {world_rank+args.data_parallel_size}")
                    dist.send(tensor=bond_output.cuda(), dst=world_rank+args.data_parallel_size)
                else:
                    target = target_lst[0]
                    target["av_task"] = target["av_task"].cuda()
                    target["bv_task"] = target["bv_task"].cuda()
                    target["fg_task"] = target["fg_task"].cuda()
                    loss = loss_func(loss, target)
                    target_lst = target_lst[1:]
                    losses.append(loss)
                if not last_iteration:
                    # Recv forward
                    if args.node_rank != 4:
                        # print(f"recv fw: {f_atoms_feature.size()} at line 2052 from {world_rank-args.data_parallel_size}")
                        f_atoms_feature = None
                        f_bonds_feature = torch.empty(f_bonds_size[i + num_warmup_microbatches + 1], args.hidden_size).cuda()
                        dist.recv(tensor=f_bonds_feature.cuda(), src=world_rank-args.data_parallel_size)
                    micro_batch = micro_batches[i + num_warmup_microbatches + 1]
                    targets = micro_batch["targets"]
                    targets["av_task"] = targets["av_task"].cuda()
                    targets["bv_task"] = targets["bv_task"].cuda()
                    targets["fg_task"] = targets["fg_task"].cuda()
                    # target_lst.append(targets)
                    micro_batch = micro_batch["graph_input"]
                    input_batch = (micro_batch[0], micro_batch[1], micro_batch[2], micro_batch[3], micro_batch[4], micro_batch[5], micro_batch[6], micro_batch[7], f_atoms_feature, f_bonds_feature)
            else:
                # Send FW recv BW
                if args.node_rank != args.model_parallel_size - 1:

                    bond_output_grad = torch.empty((f_bonds_size[i], args.hidden_size))
                    # Recv BW
                    # print(f"recv bw: {atom_output_grad.size()} at line 2054 from {world_rank + args.data_parallel_size}, micro_batch: {len(micro_batches) - i}")
                    dist.recv(tensor=bond_output_grad.cuda(), src=world_rank + args.data_parallel_size)

                    # Send FW
                    # print(f"send fw: {atom_output.size()} at line 2052 to {world_rank + args.data_parallel_size}, micro_batch: {i + num_warmup_microbatches}")
                    dist.send(tensor=bond_output.cuda(), dst=world_rank + args.data_parallel_size)

                    if f_atoms_feature is not None:
                        f_bonds_feature.retain_grad()

                    # Backward Step
                    bond_output_lst[0].backward(bond_output_grad.cuda(), retain_graph=True)
                    bond_output_lst = bond_output_lst[1:]
                else: # if last rank
                    target = target_lst[0]
                    target["av_task"] = target["av_task"].cuda()
                    target["bv_task"] = target["bv_task"].cuda()
                    target["fg_task"] = target["fg_task"].cuda()
                    loss = loss_func(loss, target)
                    target_lst = target_lst[1:]
                    losses.append(loss)
                    loss, _, _, _, _, _, _ = loss
                    loss.backward(retain_graph=True)

                if last_iteration:
                    if args.node_rank != 7:
                        # send backward
                        # print(f"send bw: {input_batches[0][-2].grad.size()} at line 2080 to {world_rank - args.data_parallel_size}")
                        dist.send(tensor=input_batches[0][-1].grad, dst=world_rank - args.data_parallel_size)
                        input_batches = input_batches[1:]
                    input_batch = None
                        
                else:
                    # Send backward
                    if args.node_rank != 7:
                        # print(f"send bw: {input_batches[0][-2].grad.size()} at line 2085 to {world_rank - args.data_parallel_size}, micro_batch: {i + num_warmup_microbatches}")
                        dist.send(tensor=input_batches[0][-1].grad, dst=world_rank - args.data_parallel_size)
                        input_batches = input_batches[1:]
                        
                    # Recv forward
                    if args.node_rank != 7:
                        f_atoms_feature = None
                        f_bonds_feature = torch.empty(f_bonds_size[i + num_warmup_microbatches + 1], args.hidden_size).cuda()
                        # print(f"recv fw: {f_atoms_feature.size()} at line 2089 from {world_rank - args.data_parallel_size}, micro_batch: {i + num_warmup_microbatches}")
                        dist.recv(tensor=f_bonds_feature.cuda(), src=world_rank - args.data_parallel_size)
                        f_bonds_feature.requires_grad = True
                    else:
                        f_atoms_feature = None
                        f_bonds_feature = None
                    micro_batch = micro_batches[i + num_warmup_microbatches + 1]
                    # targets = micro_batch["targets"]
                    micro_batch = micro_batch["graph_input"]
                    input_batch = (micro_batch[0], micro_batch[1], micro_batch[2], micro_batch[3], micro_batch[4], micro_batch[5], micro_batch[6], micro_batch[7], f_atoms_feature, f_bonds_feature)
                    input_batches.append(input_batch)
                    # if args.node_rank == 3:
                    #     targets["av_task"] = targets["av_task"].cuda()
                    #     targets["bv_task"] = targets["bv_task"].cuda()
                    #     targets["fg_task"] = targets["fg_task"].cuda()
                    #     target_lst.append(targets)
                        
        
        # cooldown phase
        # print("\nStart cooldown phase")
        if not forward_only:
            for i in range(num_warmup_microbatches):

                if i == num_warmup_microbatches - 1:
                    if args.node_rank == 0:
                        #enable_grad_sync()
                        pass
                bond_output = bond_output_lst[0]
                bond_output_grad = torch.zeros_like(bond_output)
                
                if args.node_rank != args.model_parallel_size - 1:
                    # recv backward
                    # print(f"recv bw: {atom_output_grad.size()} at line 2113 from {world_rank+args.data_parallel_size}, micro_batch: {i}")
                    dist.recv(tensor=bond_output_grad.cuda(), src=world_rank+args.data_parallel_size)
                    
                    # Backward Step
                    bond_output.backward([bond_output_grad], retain_graph=True)
                    bond_output_lst = bond_output_lst[1:]
                else:
                    loss = output_batch
                    target = target_lst[0]
                    target["av_task"] = target["av_task"].cuda()
                    target["bv_task"] = target["bv_task"].cuda()
                    target["fg_task"] = target["fg_task"].cuda()
                    loss = loss_func(loss, target)
                    target_lst = target_lst[1:]
                    losses.append(loss)
                    loss, _, _, _, _, _, _ = loss
                    loss.backward(retain_graph=True)
                if args.node_rank != 4:
                    # send backward
                    # print(f"send bw: {input_batches[0][-2].grad.size()} at line 2127 to {world_rank-args.data_parallel_size}, micro_batch: {i}")
                    dist.send(tensor=input_batches[0][-1].grad, dst=world_rank-args.data_parallel_size)
                    # release retained graph
                    input_batches = input_batches[1:]
                
        torch.cuda.empty_cache()

        # calculate loss
        if args.node_rank==7:
            if not forward_only:
                pass
        else:
            loss = None
                    
        return losses
