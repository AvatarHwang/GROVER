"""
The training function used in the finetuning task.
"""
import csv
import logging
import os
import pickle
import time
from argparse import Namespace
from logging import Logger
from typing import List

import numpy as np
import pandas as pd
import torch
from torch.optim.lr_scheduler import ExponentialLR
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
from task.predict import predict, evaluate, evaluate_predictions

import copy

import nvtx

import random
torch.manual_seed(42)
torch.cuda.manual_seed_all(42)
np.random.seed(42)
random.seed(42)
torch.backends.cudnn.deterministic = True
g = torch.Generator()
g.manual_seed(42)  

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def train(epoch, model, data, loss_func, optimizer, scheduler,
          shared_dict, args: Namespace, n_iter: int = 0,
          logger: logging.Logger = None):
    """
    Trains a model for an epoch.

    :param model: Model.
    :param data: A MoleculeDataset (or a list of MoleculeDatasets if using moe).
    :param loss_func: Loss function.
    :param optimizer: An Optimizer.
    :param scheduler: A learning rate scheduler.
    :param args: Arguments.
    :param n_iter: The number of iterations (training examples) trained on so far.
    :param logger: A logger for printing intermediate results.
    :param writer: A tensorboardX SummaryWriter.
    :return: The total number of iterations (training examples) trained on so far.
    """
    # debug = logger.debug if logger is not None else print

    model.train()

    # data.shuffle()

    loss_sum, iter_count = 0, 0
    cum_loss_sum, cum_iter_count = 0, 0


    mol_collator = MolCollator(shared_dict=shared_dict, args=args)

    num_workers = 4
    if type(data) == DataLoader:
        mol_loader = data
    else:
        mol_loader = DataLoader(data, batch_size=args.batch_size, shuffle=True,
                            num_workers=num_workers, collate_fn=mol_collator)

    for _, item in enumerate(mol_loader):
        _, batch, features_batch, mask, targets = item
        if next(model.parameters()).is_cuda:
            mask, targets = mask.cuda(), targets.cuda()
        class_weights = torch.ones(targets.shape)

        if args.cuda:
            class_weights = class_weights.cuda()

        # Run model
        model.zero_grad()
        preds = model(batch, features_batch)
        loss = loss_func(preds, targets) * class_weights * mask
        loss = loss.sum() / mask.sum()

        loss_sum += loss.item()
        iter_count += args.batch_size

        cum_loss_sum += loss.item()
        cum_iter_count += 1

        loss.backward()
        optimizer.step()

        if isinstance(scheduler, NoamLR):
            scheduler.step()

        n_iter += args.batch_size

        #if (n_iter // args.batch_size) % args.log_frequency == 0:
        #    lrs = scheduler.get_lr()
        #    loss_avg = loss_sum / iter_count
        #    loss_sum, iter_count = 0, 0
        #    lrs_str = ', '.join(f'lr_{i} = {lr:.4e}' for i, lr in enumerate(lrs))

    return n_iter, cum_loss_sum / cum_iter_count


def run_training(args: Namespace, time_start, logger: Logger = None) -> List[float]:
    """
    Trains a model and returns test scores on the model checkpoint with the highest validation score.

    :param args: Arguments.
    :param logger: Logger.
    :return: A list of ensemble scores for each task.
    """
    if logger is not None:
        debug, info = logger.debug, logger.info
    else:
        debug = info = print


    # pin GPU to local rank.
    idx = args.gpu
    if args.gpu is not None:
        torch.cuda.set_device(idx)

    features_scaler, scaler, shared_dict, test_data, train_data, val_data = load_data(args, debug, logger)

    metric_func = get_metric_func(metric=args.metric)

    # Set up test set evaluation
    test_smiles, test_targets = test_data.smiles(), test_data.targets()
    sum_test_preds = np.zeros((len(test_smiles), args.num_tasks))

    # Train ensemble of models
    for model_idx in range(args.ensemble_size):
        # Tensorboard writer
        save_dir = os.path.join(args.save_dir, f'model_{model_idx}')
        makedirs(save_dir)

        # Load/build model
        if args.checkpoint_paths is not None:
            if len(args.checkpoint_paths) == 1:
                cur_model = 0
            else:
                cur_model = model_idx
            debug(f'Loading model {cur_model} from {args.checkpoint_paths[cur_model]}')
            model = load_checkpoint(args.checkpoint_paths[cur_model], current_args=args, logger=logger)
        else:
            debug(f'Building model {model_idx}')
            model = build_model(model_idx=model_idx, args=args)

        if args.fine_tune_coff != 1 and args.checkpoint_paths is not None:
            debug("Fine tune fc layer with different lr")
            initialize_weights(model_idx=model_idx, model=model.ffn, distinct_init=args.distinct_init)

        # Get loss and metric functions
        loss_func = get_loss_func(args, model)

        optimizer = build_optimizer(model, args)

        debug(model)
        debug(f'Number of parameters = {param_count(model):,}')
        if args.cuda:
            debug('Moving model to cuda')
            model = model.cuda()

        # Ensure that model is saved in correct location for evaluation if 0 epochs
        save_checkpoint(os.path.join(save_dir, 'model.pt'), model, scaler, features_scaler, args)

        # Learning rate schedulers
        scheduler = build_lr_scheduler(optimizer, args)

        # Bulid data_loader
        shuffle = True
        mol_collator = MolCollator(shared_dict={}, args=args)
        print("train data size: ", len(train_data))
        train_data = DataLoader(train_data,
                                batch_size=args.batch_size,
                                shuffle=shuffle,
                                num_workers = 4,
                                collate_fn=mol_collator)

        # Run training
        best_score = float('inf') if args.minimize_score else -float('inf')
        best_epoch, n_iter = 0, 0
        min_val_loss = float('inf')
        for epoch in range(args.epochs):
            s_time = time.time()
            n_iter, train_loss = train(
                epoch=epoch,
                model=model,
                data=train_data,
                loss_func=loss_func,
                optimizer=optimizer,
                scheduler=scheduler,
                args=args,
                n_iter=n_iter,
                shared_dict=shared_dict,
                logger=logger
            )
            t_time = time.time() - s_time
            s_time = time.time()
            val_scores, val_loss = evaluate(
                model=model,
                data=val_data,
                loss_func=loss_func,
                num_tasks=args.num_tasks,
                metric_func=metric_func,
                batch_size=args.batch_size,
                dataset_type=args.dataset_type,
                scaler=scaler,
                shared_dict=shared_dict,
                logger=logger,
                args=args
            )
            v_time = time.time() - s_time
            # Average validation score
            avg_val_score = np.nanmean(val_scores)
            # Logged after lr step
            if isinstance(scheduler, ExponentialLR):
                scheduler.step()

            if args.show_individual_scores:
                # Individual validation scores
                for task_name, val_score in zip(args.task_names, val_scores):
                    debug(f'Validation {task_name} {args.metric} = {val_score:.6f}')
            print('Epoch: {:04d}'.format(epoch),
                  'loss_train: {:.6f}'.format(train_loss),
                  'loss_val: {:.6f}'.format(val_loss),
                  f'{args.metric}_val: {avg_val_score:.4f}',
                  # 'auc_val: {:.4f}'.format(avg_val_score),
                  'cur_lr: {:.5f}'.format(scheduler.get_lr()[-1]),
                  't_time: {:.4f}s'.format(t_time),
                  'v_time: {:.4f}s'.format(v_time))

            if args.tensorboard:
                writer.add_scalar('loss/train', train_loss, epoch)
                writer.add_scalar('loss/val', val_loss, epoch)
                writer.add_scalar(f'{args.metric}_val', avg_val_score, epoch)


            # Save model checkpoint if improved validation score
            if args.select_by_loss:
                if val_loss < min_val_loss:
                    min_val_loss, best_epoch = val_loss, epoch
                    save_checkpoint(os.path.join(save_dir, 'model.pt'), model, scaler, features_scaler, args)
            else:
                if args.minimize_score and avg_val_score < best_score or \
                        not args.minimize_score and avg_val_score > best_score:
                    best_score, best_epoch = avg_val_score, epoch
                    save_checkpoint(os.path.join(save_dir, 'model.pt'), model, scaler, features_scaler, args)

            if epoch - best_epoch > args.early_stop_epoch:
                break

        ensemble_scores = 0.0

        # Evaluate on test set using model with best validation score
        if args.select_by_loss:
            info(f'Model {model_idx} best val loss = {min_val_loss:.6f} on epoch {best_epoch}')
        else:
            info(f'Model {model_idx} best validation {args.metric} = {best_score:.6f} on epoch {best_epoch}')
        model = load_checkpoint(os.path.join(save_dir, 'model.pt'), cuda=args.cuda, logger=logger)

        test_preds, _ = predict(
            model=model,
            data=test_data,
            loss_func=loss_func,
            batch_size=args.batch_size,
            logger=logger,
            shared_dict=shared_dict,
            scaler=scaler,
            args=args
        )

        test_scores = evaluate_predictions(
            preds=test_preds,
            targets=test_targets,
            num_tasks=args.num_tasks,
            metric_func=metric_func,
            dataset_type=args.dataset_type,
            logger=logger
        )

        if len(test_preds) != 0:
            sum_test_preds += np.array(test_preds, dtype=float)

        # Average test score
        avg_test_score = np.nanmean(test_scores)
        info(f'Model {model_idx} test {args.metric} = {avg_test_score:.6f}')

        if args.show_individual_scores:
            # Individual test scores
            for task_name, test_score in zip(args.task_names, test_scores):
                info(f'Model {model_idx} test {task_name} {args.metric} = {test_score:.6f}')

        # Evaluate ensemble on test set
        avg_test_preds = (sum_test_preds / args.ensemble_size).tolist()

        ensemble_scores = evaluate_predictions(
            preds=avg_test_preds,
            targets=test_targets,
            num_tasks=args.num_tasks,
            metric_func=metric_func,
            dataset_type=args.dataset_type,
            logger=logger
        )

        ind = [['preds'] * args.num_tasks + ['targets'] * args.num_tasks, args.task_names * 2]
        ind = pd.MultiIndex.from_tuples(list(zip(*ind)))
        data = np.concatenate([np.array(avg_test_preds), np.array(test_targets)], 1)
        test_result = pd.DataFrame(data, index=test_smiles, columns=ind)
        test_result.to_csv(os.path.join(args.save_dir, 'test_result.csv'))

        # Average ensemble score
        avg_ensemble_test_score = np.nanmean(ensemble_scores)
        info(f'Ensemble test {args.metric} = {avg_ensemble_test_score:.6f}')

        # Individual ensemble scores
        if args.show_individual_scores:
            for task_name, ensemble_score in zip(args.task_names, ensemble_scores):
                info(f'Ensemble test {task_name} {args.metric} = {ensemble_score:.6f}')

    return ensemble_scores


def run_data_parallel_training(args: Namespace, time_start, logger: Logger = None) -> List[float]:
    """
    Trains a model and returns test scores on the model checkpoint with the highest validation score.

    :param args: Arguments.
    :param logger: Logger.
    :return: A list of ensemble scores for each task.
    """
    if logger is not None:
        debug, info = logger.debug, logger.info
    else:
        debug = info = print


    # pin GPU to local rank.
    rank = os.environ['LOCAL_RANK']
    world_rank = os.environ['RANK']
    torch.cuda.set_device(int(rank))

    features_scaler, scaler, shared_dict, test_data, train_data, val_data = load_data(args, debug, logger)

    metric_func = get_metric_func(metric=args.metric)

    # Set up test set evaluation
    test_smiles, test_targets = test_data.smiles(), test_data.targets()
    sum_test_preds = np.zeros((len(test_smiles), args.num_tasks))

    # Train ensemble of models
    for model_idx in range(args.ensemble_size):
        # Tensorboard writer
        if rank==0:
            save_dir = os.path.join(args.save_dir, f'model_{model_idx}')
            makedirs(save_dir)

        # Load/build model
        if args.checkpoint_paths is not None:
            if len(args.checkpoint_paths) == 1:
                cur_model = 0
            else:
                cur_model = model_idx
            debug(f'Loading model {cur_model} from {args.checkpoint_paths[cur_model]}')
            model = load_checkpoint(args.checkpoint_paths[cur_model], current_args=args, logger=logger)
        else:
            debug(f'Building model {model_idx}')
            model = build_model(model_idx=model_idx, args=args)

        if args.fine_tune_coff != 1 and args.checkpoint_paths is not None:
            debug("Fine tune fc layer with different lr")
            initialize_weights(model_idx=model_idx, model=model.ffn, distinct_init=args.distinct_init)

        # Get loss and metric functions
        loss_func = get_loss_func(args, model)

        optimizer = build_optimizer(model, args)

        debug(model)
        debug(f'Number of parameters = {param_count(model):,}')

        if args.cuda:
            debug('Moving model to cuda')
            model = model.cuda()

        model = torch.nn.parallel.DistributedDataParallel(model, find_unused_parameters=True)

        # Ensure that model is saved in correct location for evaluation if 0 epochs
        if rank==0:
            save_checkpoint(os.path.join(save_dir, 'model.pt'), model, scaler, features_scaler, args)

        # Learning rate schedulers
        scheduler = build_lr_scheduler(optimizer, args)

        # Bulid data_loader
        mol_collator = MolCollator(shared_dict={}, args=args)
        world_size = int(torch.distributed.get_world_size())
        train_sampler=torch.utils.data.distributed.DistributedSampler(train_data, num_replicas=world_size, shuffle=False)
        train_data = DataLoader(train_data,
                                batch_size=args.batch_size,
                                num_workers = 4,
                                collate_fn=mol_collator,
                                sampler=train_sampler)

        # Run training
        best_score = float('inf') if args.minimize_score else -float('inf')
        best_epoch, n_iter = 0, 0
        min_val_loss = float('inf')
        training_time = []
        loss_measure = []
        for epoch in range(args.epochs):
            s_time = time.time()
            n_iter, train_loss = train(
                epoch=epoch,
                model=model,
                data=train_data,
                loss_func=loss_func,
                optimizer=optimizer,
                scheduler=scheduler,
                args=args,
                n_iter=n_iter,
                shared_dict=shared_dict,
                logger=logger
            )
            t_time = time.time() - s_time
            training_time.append(t_time)
            s_time = time.time()
            val_scores, val_loss = evaluate(
                model=model,
                data=val_data,
                loss_func=loss_func,
                num_tasks=args.num_tasks,
                metric_func=metric_func,
                batch_size=args.batch_size,
                dataset_type=args.dataset_type,
                scaler=scaler,
                shared_dict=shared_dict,
                logger=logger,
                args=args
            )
            v_time = time.time() - s_time
            # Average validation score
            avg_val_score = np.nanmean(val_scores)
            # Logged after lr step
            if isinstance(scheduler, ExponentialLR):
                scheduler.step()

            if args.show_individual_scores:
                # Individual validation scores
                for task_name, val_score in zip(args.task_names, val_scores):
                    debug(f'Validation {task_name} {args.metric} = {val_score:.6f}')
            loss_measure.append(val_loss)
            print('Epoch: {:04d}'.format(epoch),
                'loss_train: {:.6f}'.format(train_loss),
                'loss_val: {:.6f}'.format(val_loss),
                f'{args.metric}_val: {avg_val_score:.4f}',
                # 'auc_val: {:.4f}'.format(avg_val_score),
                'cur_lr: {:.5f}'.format(scheduler.get_lr()[-1]),
                't_time: {:.4f}s'.format(t_time),
                'v_time: {:.4f}s'.format(v_time))

            if args.tensorboard:
                writer.add_scalar('loss/train', train_loss, epoch)
                writer.add_scalar('loss/val', val_loss, epoch)
                writer.add_scalar(f'{args.metric}_val', avg_val_score, epoch)


            # Save model checkpoint if improved validation score
            if args.select_by_loss:
                if val_loss < min_val_loss:
                    min_val_loss, best_epoch = val_loss, epoch
                    if rank==0:
                        save_checkpoint(os.path.join(save_dir, 'model.pt'), model, scaler, features_scaler, args)
            else:
                if args.minimize_score and avg_val_score < best_score or \
                        not args.minimize_score and avg_val_score > best_score:
                    best_score, best_epoch = avg_val_score, epoch
                    if rank==0:
                        save_checkpoint(os.path.join(save_dir, 'model.pt'), model, scaler, features_scaler, args)

            if epoch - best_epoch > args.early_stop_epoch:
                break
           
        ensemble_scores = 0.0

        # Evaluate on test set using model with best validation score
        if args.select_by_loss:
            info(f'Model {model_idx} best val loss = {min_val_loss:.6f} on epoch {best_epoch}')
        else:
            info(f'Model {model_idx} best validation {args.metric} = {best_score:.6f} on epoch {best_epoch}')
        dist.barrier()
        if rank==0:
            model = load_checkpoint(os.path.join(save_dir, 'model.pt'), cuda=args.cuda, logger=logger)

        test_preds, _ = predict(
            model=model,
            data=test_data,
            loss_func=loss_func,
            batch_size=args.batch_size,
            logger=logger,
            shared_dict=shared_dict,
            scaler=scaler,
            args=args
        )

        test_scores = evaluate_predictions(
            preds=test_preds,
            targets=test_targets,
            num_tasks=args.num_tasks,
            metric_func=metric_func,
            dataset_type=args.dataset_type,
            logger=logger
        )

        if len(test_preds) != 0:
            sum_test_preds += np.array(test_preds, dtype=float)

        # Average test score
        avg_test_score = np.nanmean(test_scores)
        info(f'Model {model_idx} test {args.metric} = {avg_test_score:.6f}')

        if args.show_individual_scores:
            # Individual test scores
            for task_name, test_score in zip(args.task_names, test_scores):
                info(f'Model {model_idx} test {task_name} {args.metric} = {test_score:.6f}')

        # Evaluate ensemble on test set
        avg_test_preds = (sum_test_preds / args.ensemble_size).tolist()

        ensemble_scores = evaluate_predictions(
            preds=avg_test_preds,
            targets=test_targets,
            num_tasks=args.num_tasks,
            metric_func=metric_func,
            dataset_type=args.dataset_type,
            logger=logger
        )

        ind = [['preds'] * args.num_tasks + ['targets'] * args.num_tasks, args.task_names * 2]
        ind = pd.MultiIndex.from_tuples(list(zip(*ind)))
        data = np.concatenate([np.array(avg_test_preds), np.array(test_targets)], 1)
        test_result = pd.DataFrame(data, index=test_smiles, columns=ind)
        if rank==0:
            test_result.to_csv(os.path.join(args.save_dir, 'test_result.csv'))

        # Average ensemble score
        avg_ensemble_test_score = np.nanmean(ensemble_scores)
        info(f'Ensemble test {args.metric} = {avg_ensemble_test_score:.6f}')

        # Individual ensemble scores
        if args.show_individual_scores:
            for task_name, ensemble_score in zip(args.task_names, ensemble_scores):
                info(f'Ensemble test {task_name} {args.metric} = {ensemble_score:.6f}')
        print("training time:", training_time)

        return ensemble_scores


def run_task_parallel_training(args: Namespace, time_start, logger: Logger = None) -> List[float]:
    """
    Trains a model and returns test scores on the model checkpoint with the highest validation score.

    :param args: Arguments.
    :param logger: Logger.
    :return: A list of ensemble scores for each task.
    """

    from ParallelModelVersion2 import Node_Block_parallel as model0
    from ParallelModelVersion2 import Edge_Block_parallel as model1
    from ParallelModelVersion2 import NodeViewReadoutFFN as model2
    from ParallelModelVersion2 import EdgeViewReadoutFFN as model3

    local_rank = int(os.environ['LOCAL_RANK'])
    world_rank = int(os.environ['RANK'])
    torch.cuda.set_device(local_rank)
    tag_id = int((local_rank)*100)

    if logger is not None:
        debug, info = logger.debug, logger.info
    else:
        debug = info = print

    world_size = args.world_size
    group_node_ranks=[i for i in range(world_size//2)]
    group_edge_ranks=[i for i in range(world_size//2, world_size)]
    group_node = dist.new_group(group_node_ranks)
    group_edge = dist.new_group(group_edge_ranks) 

    num_workers = 4

    features_scaler, scaler, shared_dict, test_data, train_data, val_data = load_data(args, debug, logger)

    metric_func = get_metric_func(metric=args.metric)

    # Set up test set evaluation
    test_smiles, test_targets = test_data.smiles(), test_data.targets()
    sum_test_preds = np.zeros((len(test_smiles), args.num_tasks))

    # Train ensemble of models
    for model_idx in range(args.ensemble_size):
        # Tensorboard writer
        save_dir = os.path.join(args.save_dir, f'model_{model_idx}')
        makedirs(save_dir)

        # Load/build model
        if args.checkpoint_paths is not None:
            if len(args.checkpoint_paths) == 1:
                cur_model = 0
            else:
                cur_model = model_idx
            debug(f'Loading model {cur_model} from {args.checkpoint_paths[cur_model]}')
            model = load_checkpoint(args.checkpoint_paths[cur_model], current_args=args, logger=logger)
        else:
            debug(f'Building model {model_idx}')
            model = build_model(model_idx=model_idx, args=args)

        if args.fine_tune_coff != 1 and args.checkpoint_paths is not None:
            debug("Fine tune fc layer with different lr")
            initialize_weights(model_idx=model_idx, model=model.ffn, distinct_init=args.distinct_init)

        # Get loss and metric functions
        loss_func = get_loss_func(args, model)

        if world_rank < world_size//2:
            training_time = []
            model_0 = model0(model=model, rank=local_rank).cuda()
            model_2 = model2(model=model, rank=local_rank, args=args).cuda()

            if world_size > 2:
                model_0 = DDP(model_0, process_group=group_node)
                model_2 = DDP(model_2, process_group=group_node, find_unused_parameters=True)

            optimizer_0 = build_optimizer(model_0, args)
            optimizer_2 = build_optimizer(model_2, args)

            # Ensure that model is saved in correct location for evaluation if 0 epochs
            if local_rank==0:
                torch.save(model_0.state_dict(), os.path.join(save_dir, "model0.pt"))
                torch.save(model_2.state_dict(), os.path.join(save_dir, "model2.pt"))

            # Learning rate schedulers
            scheduler_0 = build_lr_scheduler(optimizer_0, args)
            scheduler_2 = build_lr_scheduler(optimizer_2, args)

            # Bulid data_loader
            shuffle = True
            mol_collator = MolCollator(shared_dict={}, args=args)
            print("train data size: ", len(train_data))

            if world_size>2:
                train_sampler=torch.utils.data.distributed.DistributedSampler(train_data, num_replicas=int(world_size/2), shuffle=False)
                train_data = DataLoader(train_data,
                                        batch_size=args.batch_size,
                                        num_workers=num_workers,
                                        drop_last=True,
                                        sampler=train_sampler,
                                        collate_fn=mol_collator,
                                        pin_memory=True)
            else:
                train_data = DataLoader(train_data,
                                        batch_size=args.batch_size,
                                        shuffle=False,
                                        num_workers = 4,
                                        worker_init_fn=seed_worker,
                                        generator=g,
                                        collate_fn=mol_collator,
                                        pin_memory=True)

            # Run training
            best_score = float('inf') if args.minimize_score else -float('inf')
            best_epoch, n_iter = 0, 0
            min_val_loss = float('inf')
            for epoch in range(args.epochs):
                s_time = time.time()
                n_iter, train_loss = task_parallel_train(
                epoch=epoch,
                model_0=model_0,
                model_2=model_2, 
                data=train_data, 
                loss_func=loss_func, 
                optimizer_0=optimizer_0, 
                optimizer_2=optimizer_2, 
                scheduler_0=scheduler_0, 
                scheduler_2=scheduler_2, 
                shared_dict=shared_dict, 
                args=args, 
                n_iter = n_iter, 
                logger = logger,
                num_workers=num_workers)
                t_time = time.time() - s_time
                training_time.append(t_time)
                s_time = time.time()
                val_scores, val_loss = task_parallel_evaluate(model_0=model_0,
                                                model_2=model_2,
                                                data=val_data,
                                                loss_func=loss_func,
                                                num_tasks=args.num_tasks,
                                                metric_func=metric_func,
                                                batch_size=args.batch_size,
                                                dataset_type=args.dataset_type,
                                                scaler=scaler,
                                                shared_dict=shared_dict,
                                                logger=logger,
                                                args=args
                                                )
                v_time = time.time() - s_time
                # Average validation score
                avg_val_score = np.nanmean(val_scores)
                # Logged after lr step
                if isinstance(scheduler_0, ExponentialLR):
                    scheduler_0.step()
                if isinstance(scheduler_2, ExponentialLR):
                    scheduler_2.step()

                if args.show_individual_scores:
                    # Individual validation scores
                    for task_name, val_score in zip(args.task_names, val_scores):
                        debug(f'Validation {task_name} {args.metric} = {val_score:.6f}')
                if local_rank==0:
                    print('Epoch: {:04d}'.format(epoch),
                        'loss_train: {:.6f}'.format(train_loss),
                        'loss_val: {:.6f}'.format(val_loss),
                        f'{args.metric}_val: {avg_val_score:.4f}',
                        # 'auc_val: {:.4f}'.format(avg_val_score),
                        'cur_lr_0: {:.5f}'.format(scheduler_0.get_lr()[-1]),
                        'cur_lr_2: {:.5f}'.format(scheduler_2.get_lr()[-1]),
                        't_time: {:.4f}s'.format(t_time),
                        'v_time: {:.4f}s'.format(v_time))

                if args.tensorboard:
                    writer.add_scalar('loss/train', train_loss, epoch)
                    writer.add_scalar('loss/val', val_loss, epoch)
                    writer.add_scalar(f'{args.metric}_val', avg_val_score, epoch)

            # Save model checkpoint if improved validation score
            if local_rank==0:
                if args.select_by_loss:
                    if val_loss < min_val_loss:
                        min_val_loss, best_epoch = val_loss, epoch
                        save_checkpoint(os.path.join(save_dir, 'model.pt'), model, scaler, features_scaler, args)
                else:
                    if args.minimize_score and avg_val_score < best_score or \
                            not args.minimize_score and avg_val_score > best_score:
                        best_score, best_epoch = avg_val_score, epoch
                        save_checkpoint(os.path.join(save_dir, 'model.pt'), model, scaler, features_scaler, args)

            if epoch - best_epoch > args.early_stop_epoch:
                break

        else:
            model_1 = model1(model=model, rank=local_rank).cuda() # Edge_blocks
            model_3 = model3(model=model, rank=local_rank, args=args).cuda()
            if world_size > 2:
                model_1 = DDP(model_1, process_group=group_edge)
                model_3 = DDP(model_3, process_group=group_edge, find_unused_parameters=True)

            optimizer_1 = build_optimizer(model_1, args)
            optimizer_3 = build_optimizer(model_3, args)

            # Build learning rate scheduler
            scheduler_1 = build_lr_scheduler(optimizer_1, args)
            scheduler_3 = build_lr_scheduler(optimizer_3, args)

            # Set up DataLoader
            mol_collator = MolCollator(shared_dict={}, args=args)

            if world_size>2:
                train_sampler=torch.utils.data.distributed.DistributedSampler(train_data, rank=int(world_rank%(world_size/2)), num_replicas=int(world_size/2), shuffle=False)
                train_data = DataLoader(train_data,
                                        batch_size=args.batch_size,
                                        num_workers = 4,
                                        drop_last=True,
                                        sampler=train_sampler,
                                        collate_fn=mol_collator,
                                        pin_memory=True)

            best_score = float('inf') if args.minimize_score else -float('inf')
            best_epoch, n_iter = 0, 0
            min_val_loss = float('inf')

            for epoch in range(args.epochs):
                n_iter, train_loss = train_rank1(
                    epoch=epoch,
                    model_1=model_1,
                    model_3=model_3, 
                    data=train_data, 
                    loss_func=loss_func, 
                    optimizer_0=optimizer_1, 
                    optimizer_2=optimizer_3, 
                    scheduler_0=scheduler_1, 
                    scheduler_2=scheduler_3, 
                    shared_dict=shared_dict, 
                    args=args, 
                    n_iter = n_iter, 
                    logger = logger,
                    num_workers=num_workers)

                val_loss = evaluate_rank1(model_1, model_3, val_data, loss_func, args, shared_dict)

                # Save model checkpoint if improved validation score
                if local_rank==0:
                    if args.select_by_loss:
                        if val_loss < min_val_loss:
                            min_val_loss, best_epoch = val_loss, epoch
                            torch.save(model_1.state_dict(), os.path.join(save_dir, "model1.pt"))
                            torch.save(model_3.state_dict(), os.path.join(save_dir, "model3.pt"))
                    else:
                        if args.minimize_score and avg_val_score < best_score or \
                                not args.minimize_score and avg_val_score > best_score:
                            best_score, best_epoch = avg_val_score, epoch
                            torch.save(model_1.state_dict(), os.path.join(save_dir, "model1.pt"))
                            torch.save(model_3.state_dict(), os.path.join(save_dir, "model3.pt"))

        if world_rank==0:
            ensemble_scores = 0.0

            # Evaluate on test set using model with best validation score
            if args.select_by_loss:
                info(f'Model best val loss = {min_val_loss:.6f} on epoch {best_epoch}')
            else:
                info(f'Model best validation {args.metric} = {best_score:.6f} on epoch {best_epoch}')
            model_0.load_state_dict(torch.load(os.path.join(save_dir, "model0.pt")))
            model_2.load_state_dict(torch.load(os.path.join(save_dir, "model2.pt")))

            # Send best epoch to rank 1
            
            test_preds, _ = task_parallel_predict(
                model_0=model_0,
                model_2=model_2,
                data=test_data,
                loss_func=loss_func,
                batch_size=args.batch_size,
                logger=logger,
                shared_dict=shared_dict,
                scaler=scaler,
                args=args
            )

            test_scores = evaluate_predictions(
                preds=test_preds,
                targets=test_targets,
                num_tasks=args.num_tasks,
                metric_func=metric_func,
                dataset_type=args.dataset_type,
                logger=logger
            )

            if len(test_preds) != 0:
                sum_test_preds += np.array(test_preds, dtype=float)

            # Average test score
            avg_test_score = np.nanmean(test_scores)
            print(f"test scores:{test_scores}")
            info(f'Model test {args.metric} = {avg_test_score:.6f}')

            if args.show_individual_scores:
                # Individual test scores
                for task_name, test_score in zip(args.task_names, test_scores):
                    info(f'Model test {task_name} {args.metric} = {test_score:.6f}')

            # Evaluate ensemble on test set
            avg_test_preds = (sum_test_preds / args.ensemble_size).tolist()

            ensemble_scores = evaluate_predictions(
                preds=avg_test_preds,
                targets=test_targets,
                num_tasks=args.num_tasks,
                metric_func=metric_func,
                dataset_type=args.dataset_type,
                logger=logger
            )

            ind = [['preds'] * args.num_tasks + ['targets'] * args.num_tasks, args.task_names * 2]
            ind = pd.MultiIndex.from_tuples(list(zip(*ind)))
            data = np.concatenate([np.array(avg_test_preds), np.array(test_targets)], 1)
            test_result = pd.DataFrame(data, index=test_smiles, columns=ind)
            test_result.to_csv(os.path.join(args.save_dir, 'test_result.csv'))

            # Average ensemble score
            avg_ensemble_test_score = np.nanmean(ensemble_scores)
            info(f'Ensemble test {args.metric} = {avg_ensemble_test_score:.6f}')

            # Individual ensemble scores
            if args.show_individual_scores:
                for task_name, ensemble_score in zip(args.task_names, ensemble_scores):
                    info(f'Ensemble test {task_name} {args.metric} = {ensemble_score:.6f}')
            print(f"ensemble_scores : {ensemble_scores}")

            #Delete: for measurement
            print(f"\ntraining time is:{training_time}")
            #print(f"\nloss_measure time is:{loss_measure}")

            return ensemble_scores

        if world_rank==int(dist.get_world_size()/2):
            # Load the best model
            model_1.load_state_dict(torch.load(os.path.join(save_dir, "model1.pt")))
            model_3.load_state_dict(torch.load(os.path.join(save_dir, "model3.pt")))

            model_1.eval()
            model_3.eval()
            val_mol_collator = MolCollator(args=args, shared_dict=shared_dict)
            num_workers = 4
            val_mol_loader = DataLoader(test_data, batch_size=args.batch_size, shuffle=False, num_workers=num_workers,worker_init_fn=seed_worker,
                                        generator=g, collate_fn=val_mol_collator)
            for _, item in enumerate(val_mol_loader):
                _, batch, features_batch2, mask, targets = item

                with torch.no_grad():
                    f_atoms, f_bonds, a2b, b2a, b2revb, a_scope, b_scope, a2a = batch
                    bond_output = model_1(f_atoms.cuda(), f_bonds.cuda(), a2b.cuda(), b2a.cuda(), b2revb.cuda(), a_scope.cuda(), b_scope.cuda(), a2a.cuda(), features_batch2)

                    atom_output = torch.zeros(f_atoms.size(0), args.hidden_size).cuda() # elements will be changed in forward

                    preds = model_3(atom_output, bond_output, f_atoms, a2b, a_scope, features_batch2)
            #


def run_pp_training(args: Namespace, time_start, logger: Logger = None) -> List[float]:

    local_rank = int(os.environ['LOCAL_RANK'])
    world_rank = int(os.environ['RANK'])
    torch.cuda.set_device(local_rank)

    if logger is not None:
        debug, info = logger.debug, logger.info
    else:
        debug = info = print

    world_size = args.world_size
    if world_size > 4:
        my_dp_group = [i for i in range(args.node_rank*4, args.node_rank*4+4)] 
        my_dp_group = dist.new_group(my_dp_group)

    num_workers = 4

    features_scaler, scaler, shared_dict, test_data, train_data, val_data = load_data(args, debug, logger)

    metric_func = get_metric_func(metric=args.metric)

    # Set up test set evaluation
    test_smiles, test_targets = test_data.smiles(), test_data.targets()
    sum_test_preds = np.zeros((len(test_smiles), args.num_tasks))

    # Train ensemble of models
    for model_idx in range(args.ensemble_size):
        # Tensorboard writer
        save_dir = os.path.join(args.save_dir, f'model_{model_idx}')
        makedirs(save_dir)

        # Load/build model
        if args.checkpoint_paths is not None:
            if len(args.checkpoint_paths) == 1:
                cur_model = 0
            else:
                cur_model = model_idx
            debug(f'Loading model {cur_model} from {args.checkpoint_paths[cur_model]}')
            model = load_checkpoint(args.checkpoint_paths[cur_model], current_args=args, logger=logger)
        else:
            debug(f'Building model {model_idx}')
            model = build_model(model_idx=model_idx, args=args)

        if args.fine_tune_coff != 1 and args.checkpoint_paths is not None:
            debug("Fine tune fc layer with different lr")
            initialize_weights(model_idx=model_idx, model=model.ffn, distinct_init=args.distinct_init)

        model = model.cuda()
        # Get loss and metric functions
        loss_func = get_loss_func(args, model)

        training_time = []

        if world_size > 4:
            model = DDP(model, process_group=my_dp_group, find_unused_parameters=True)

        optimizer = build_optimizer(model, args)

        # Ensure that model is saved in correct location for evaluation if 0 epochs
        if local_rank==0:
            torch.save(model.state_dict(), os.path.join(save_dir, f"model_{args.node_rank}.pt"))

        # Learning rate schedulers
        scheduler = build_lr_scheduler(optimizer, args)

        shuffle = True
        mol_collator = MolCollator(shared_dict={}, args=args)
        print("train data size: ", len(train_data))

        if world_size>4:
            train_sampler=torch.utils.data.distributed.DistributedSampler(train_data, num_replicas=4, rank=local_rank, shuffle=False)
            train_data = DataLoader(train_data,
                                    batch_size=args.batch_size,
                                    num_workers=num_workers,
                                    drop_last=True,
                                    sampler=train_sampler,
                                    collate_fn=mol_collator,
                                    pin_memory=True)
        else:
            train_data = DataLoader(train_data,
                                    batch_size=args.batch_size,
                                    shuffle=False,
                                    num_workers = 4,
                                    worker_init_fn=seed_worker,
                                    generator=g,
                                    collate_fn=mol_collator,
                                    pin_memory=True)

        # Run training
        best_score = float('inf') if args.minimize_score else -float('inf')
        best_epoch, n_iter = 0, 0
        min_val_loss = float('inf')
        for epoch in range(args.epochs):
            s_time = time.time()
            n_iter, train_loss = train_pp(
            epoch=epoch,
            model=model,
            data=train_data, 
            loss_func=loss_func, 
            optimizer=optimizer, 
            scheduler=scheduler, 
            shared_dict=shared_dict, 
            args=args, 
            n_iter = n_iter, 
            logger = logger,
            num_workers=num_workers)
            t_time = time.time() - s_time
            training_time.append(t_time)
            s_time = time.time()
            val_scores, val_loss = evaluate(
                model=model,
                data=val_data,
                loss_func=loss_func,
                num_tasks=args.num_tasks,
                metric_func=metric_func,
                batch_size=args.batch_size,
                dataset_type=args.dataset_type,
                scaler=scaler,
                shared_dict=shared_dict,
                logger=logger,
                args=args
            )
            v_time = time.time() - s_time
            # Average validation score
            avg_val_score = np.nanmean(val_scores)
            # Logged after lr step
            if isinstance(scheduler, ExponentialLR):
                scheduler.step()

            if args.show_individual_scores:
                # Individual validation scores
                for task_name, val_score in zip(args.task_names, val_scores):
                    debug(f'Validation {task_name} {args.metric} = {val_score:.6f}')
            if local_rank==0:
                print('Epoch: {:04d}'.format(epoch),
                    'loss_train: {:.6f}'.format(train_loss),
                    'loss_val: {:.6f}'.format(val_loss),
                    f'{args.metric}_val: {avg_val_score:.4f}',
                    # 'auc_val: {:.4f}'.format(avg_val_score),
                    'cur_lr_0: {:.5f}'.format(scheduler_0.get_lr()[-1]),
                    'cur_lr_2: {:.5f}'.format(scheduler_2.get_lr()[-1]),
                    't_time: {:.4f}s'.format(t_time),
                    'v_time: {:.4f}s'.format(v_time))

            if args.tensorboard:
                writer.add_scalar('loss/train', train_loss, epoch)
                writer.add_scalar('loss/val', val_loss, epoch)
                writer.add_scalar(f'{args.metric}_val', avg_val_score, epoch)

        # Save model checkpoint if improved validation score
        if local_rank==0:
            if args.select_by_loss:
                if val_loss < min_val_loss:
                    min_val_loss, best_epoch = val_loss, epoch
                    save_checkpoint(os.path.join(save_dir, 'model.pt'), model, scaler, features_scaler, args)
            else:
                if args.minimize_score and avg_val_score < best_score or \
                        not args.minimize_score and avg_val_score > best_score:
                    best_score, best_epoch = avg_val_score, epoch
                    save_checkpoint(os.path.join(save_dir, 'model.pt'), model, scaler, features_scaler, args)

        if epoch - best_epoch > args.early_stop_epoch:
                break

        if world_rank==0:
            ensemble_scores = 0.0

            # Evaluate on test set using model with best validation score
            if args.select_by_loss:
                info(f'Model best val loss = {min_val_loss:.6f} on epoch {best_epoch}')
            else:
                info(f'Model best validation {args.metric} = {best_score:.6f} on epoch {best_epoch}')
            model.load_state_dict(torch.load(os.path.join(save_dir, "model.pt")))

            # Send best epoch to rank 1
            
            test_preds, _ = task_parallel_predict(
                model_0=model_0,
                model_2=model_2,
                data=test_data,
                loss_func=loss_func,
                batch_size=args.batch_size,
                logger=logger,
                shared_dict=shared_dict,
                scaler=scaler,
                args=args
            )

            test_scores = evaluate_predictions(
                preds=test_preds,
                targets=test_targets,
                num_tasks=args.num_tasks,
                metric_func=metric_func,
                dataset_type=args.dataset_type,
                logger=logger
            )

            if len(test_preds) != 0:
                sum_test_preds += np.array(test_preds, dtype=float)

            # Average test score
            avg_test_score = np.nanmean(test_scores)
            print(f"test scores:{test_scores}")
            info(f'Model test {args.metric} = {avg_test_score:.6f}')

            if args.show_individual_scores:
                # Individual test scores
                for task_name, test_score in zip(args.task_names, test_scores):
                    info(f'Model test {task_name} {args.metric} = {test_score:.6f}')

            # Evaluate ensemble on test set
            avg_test_preds = (sum_test_preds / args.ensemble_size).tolist()

            ensemble_scores = evaluate_predictions(
                preds=avg_test_preds,
                targets=test_targets,
                num_tasks=args.num_tasks,
                metric_func=metric_func,
                dataset_type=args.dataset_type,
                logger=logger
            )

            ind = [['preds'] * args.num_tasks + ['targets'] * args.num_tasks, args.task_names * 2]
            ind = pd.MultiIndex.from_tuples(list(zip(*ind)))
            data = np.concatenate([np.array(avg_test_preds), np.array(test_targets)], 1)
            test_result = pd.DataFrame(data, index=test_smiles, columns=ind)
            test_result.to_csv(os.path.join(args.save_dir, 'test_result.csv'))

            # Average ensemble score
            avg_ensemble_test_score = np.nanmean(ensemble_scores)
            info(f'Ensemble test {args.metric} = {avg_ensemble_test_score:.6f}')

            # Individual ensemble scores
            if args.show_individual_scores:
                for task_name, ensemble_score in zip(args.task_names, ensemble_scores):
                    info(f'Ensemble test {task_name} {args.metric} = {ensemble_score:.6f}')
            print(f"ensemble_scores : {ensemble_scores}")

            #Delete: for measurement
            print(f"\ntraining time is:{training_time}")
            #print(f"\nloss_measure time is:{loss_measure}")

            return ensemble_scores

        if world_rank==int(dist.get_world_size()/2):
            # Load the best model
            model_1.load_state_dict(torch.load(os.path.join(save_dir, "model1.pt")))
            model_3.load_state_dict(torch.load(os.path.join(save_dir, "model3.pt")))

            model_1.eval()
            model_3.eval()
            val_mol_collator = MolCollator(args=args, shared_dict=shared_dict)
            num_workers = 4
            val_mol_loader = DataLoader(test_data, batch_size=args.batch_size, shuffle=False, num_workers=num_workers,worker_init_fn=seed_worker,
                                        generator=g, collate_fn=val_mol_collator)
            for _, item in enumerate(val_mol_loader):
                _, batch, features_batch2, mask, targets = item

                with torch.no_grad():
                    f_atoms, f_bonds, a2b, b2a, b2revb, a_scope, b_scope, a2a = batch
                    bond_output = model_1(f_atoms.cuda(), f_bonds.cuda(), a2b.cuda(), b2a.cuda(), b2revb.cuda(), a_scope.cuda(), b_scope.cuda(), a2a.cuda(), features_batch2)

                    atom_output = torch.zeros(f_atoms.size(0), args.hidden_size).cuda() # elements will be changed in forward

                    preds = model_3(atom_output, bond_output, f_atoms, a2b, a_scope, features_batch2)


def task_parallel_train(epoch, model_0, model_2, data, loss_func, optimizer_0, optimizer_2, scheduler_0,
          scheduler_2, shared_dict, args, n_iter = 0,
          logger = None, num_workers = 4):

    model_0.train()
    model_2.train()

    loss_sum, iter_count = 0, 0
    cum_loss_sum, cum_iter_count = 0, 0

    mol_collator = MolCollator(shared_dict=shared_dict, args=args)

    #num_workers = 4
    if type(data) == DataLoader:
        mol_loader = data
    else:
        mol_loader = DataLoader(data, batch_size=args.batch_size, shuffle=False,
                            num_workers=num_workers, collate_fn=mol_collator, pin_memory=True)

    """
    with torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA
        ],
        schedule=torch.profiler.schedule(wait=0, warmup=40, active=10),
        on_trace_ready=torch.profiler.tensorboard_trace_handler('./log'),
        with_stack=True,
        with_flops=True
        ) as p:
    """
    for _, item in enumerate(mol_loader):
        with nvtx.annotate(f"rank {dist.get_rank()} step {n_iter/args.batch_size}"):
            step_time = time.time()
            _, batch, features_batch, mask, targets = item
            f_atoms, f_bonds, a2b, b2a, b2revb, a_scope, b_scope, a2a = batch

            if next(model_0.parameters()).is_cuda:
                mask, targets = mask.cuda(), targets.cuda()
            class_weights = torch.ones(targets.shape).cuda()

            with nvtx.annotate(f"zerograd even {n_iter/args.batch_size}", color="red"):
                model_0.zero_grad()
                model_2.zero_grad()
            
            with nvtx.annotate(f"model0 even {n_iter/args.batch_size}", color="orange"):
                atom_output = model_0(f_atoms, f_bonds, a2b, b2a, b2revb, a_scope, b_scope, a2a, features_batch)
            
            with nvtx.annotate(f"model2 even {n_iter/args.batch_size}", color="yellow"):
                preds = model_2(atom_output, f_atoms, a2a, a_scope, features_batch)
            
            loss = loss_func(preds, targets) * class_weights * mask
            loss = loss.sum() / mask.sum()

            loss_sum += loss.item()
            iter_count += args.batch_size

            cum_loss_sum += loss.item()
            cum_iter_count += 1

            with nvtx.annotate(f"backward even {n_iter/args.batch_size}", color="green"):
                loss.backward()

            with nvtx.annotate(f"optim even {n_iter/args.batch_size}", color="blue"):
                optimizer_2.step()
                if isinstance(scheduler_2, NoamLR):
                    scheduler_2.step()
                optimizer_0.step()
                if isinstance(scheduler_0, NoamLR):
                    scheduler_0.step()

            n_iter += args.batch_size

                    #p.step()

    return n_iter, (cum_loss_sum / cum_iter_count)


def task_parallel_evaluate(model_0,
             model_2,
             data,
             num_tasks: int,
             metric_func,
             loss_func,
             batch_size: int,
             dataset_type: str,
             args,
             shared_dict,
             scaler = None,
             logger = None):

    model_0.eval()
    model_2.eval()
    args.bond_drop_rate = 0
    preds = []
    # num_iters, iter_step = len(data), batch_size
    loss_sum, iter_count = 0, 0
    mol_collator = MolCollator(args=args, shared_dict=shared_dict)
    num_workers = 4
    mol_loader = DataLoader(data, batch_size=args.batch_size, shuffle=False, num_workers=num_workers, worker_init_fn=seed_worker,
                                generator=g, collate_fn=mol_collator)
    for _, item in enumerate(mol_loader):
        _, batch, features_batch, mask, targets = item
        class_weights = torch.ones(targets.shape)
        if next(model_0.parameters()).is_cuda:
            targets = targets.cuda()
            mask = mask.cuda()
            class_weights = class_weights.cuda()
        with torch.no_grad():
            f_atoms, f_bonds, a2b, b2a, b2revb, a_scope, b_scope, a2a = batch
            
            atom_output = model_0(f_atoms, f_bonds, a2b, b2a, b2revb, a_scope, b_scope, a2a, features_batch)

            batch_preds = model_2(atom_output, f_atoms, a2a, a_scope, features_batch)
            
            iter_count += 1
            if args.fingerprint:
                batch_preds.extend(batch_preds.data.cpu().numpy())
                continue

            if loss_func is not None:
                loss = loss_func(batch_preds, targets) * class_weights * mask
                loss = loss.sum() / mask.sum()
                loss_sum += loss.item()
        # Collect vectors
        batch_preds = batch_preds.data.cpu().numpy().tolist()
        if scaler is not None:
            batch_preds = scaler.inverse_transform(batch_preds)
        preds.extend(batch_preds)
    loss_avg = loss_sum / iter_count

    targets = data.targets()
    if scaler is not None:
        targets = scaler.inverse_transform(targets)

    targets = data.targets()
    if scaler is not None:
        targets = scaler.inverse_transform(targets)

    results = evaluate_predictions(
        preds=preds,
        targets=targets,
        num_tasks=num_tasks,
        metric_func=metric_func,
        dataset_type=dataset_type,
        logger=logger
    )

    return results, loss_avg


def task_parallel_predict(model_0,
        model_2,
        data,
        args,
        batch_size: int,
        loss_func,
        logger,
        shared_dict,
        scaler = None
        ):

    # debug = logger.debug if logger is not None else print
    model_0.eval()
    model_2.eval()
    args.bond_drop_rate = 0
    preds = []

    # num_iters, iter_step = len(data), batch_size
    loss_sum, iter_count = 0, 0

    mol_collator = MolCollator(args=args, shared_dict=shared_dict)
    # mol_dataset = MoleculeDataset(data)

    num_workers = 4
    mol_loader = DataLoader(data, batch_size=batch_size, shuffle=False, num_workers=num_workers,worker_init_fn=seed_worker,
                            generator=g, collate_fn=mol_collator)
    for _, item in enumerate(mol_loader):
        _, batch, features_batch, mask, targets = item
        class_weights = torch.ones(targets.shape)
        if next(model_0.parameters()).is_cuda:
            targets = targets.cuda()
            mask = mask.cuda()
            class_weights = class_weights.cuda()
        with torch.no_grad():
            f_atoms, f_bonds, a2b, b2a, b2revb, a_scope, b_scope, a2a = batch
            atom_output = model_0(f_atoms, f_bonds, a2b, b2a, b2revb, a_scope, b_scope, a2a, features_batch)

            batch_preds = model_2(atom_output,f_atoms, a2a, a_scope, features_batch)

            iter_count += 1
            if args.fingerprint:
                preds.extend(batch_preds.data.cpu().numpy())
                continue

            if loss_func is not None:
                loss = loss_func(batch_preds, targets) * class_weights * mask
                loss = loss.sum() / mask.sum()
                loss_sum += loss.item()
        # Collect vectors
        batch_preds = batch_preds.data.cpu().numpy().tolist()
        if scaler is not None:
            batch_preds = scaler.inverse_transform(batch_preds)
        preds.extend(batch_preds)

    loss_avg = loss_sum / iter_count
    return preds, loss_avg


def train_rank1(epoch, model_1, model_3, data, loss_func, optimizer_0, optimizer_2, scheduler_0,
          scheduler_2, shared_dict, args, n_iter = 0,
          logger = None, num_workers = 4):

    rank = int(os.environ['RANK'])
    
    model_1.train()
    model_3.train()

    loss_sum, iter_count = 0, 0
    cum_loss_sum, cum_iter_count = 0, 0

    mol_collator = MolCollator(shared_dict=shared_dict, args=args)

    #num_workers = 4
    if type(data) == DataLoader:
        mol_loader = data
    else:
        mol_loader = DataLoader(data, batch_size=args.batch_size, shuffle=False,
                            num_workers=num_workers, collate_fn=mol_collator, pin_memory=True)
    """
    with torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA
        ],
        schedule=torch.profiler.schedule(wait=0, warmup=40, active=10),
        on_trace_ready=torch.profiler.tensorboard_trace_handler('./log'),
        with_stack=True,
        with_flops=True
        ) as p:
    """
    for _, item in enumerate(mol_loader):
        with nvtx.annotate(f"rank {dist.get_rank()} step {n_iter/args.batch_size}"):
            _, batch, features_batch, mask, targets = item
            f_atoms, f_bonds, a2b, b2a, b2revb, a_scope, b_scope, a2a = batch

            if next(model_1.parameters()).is_cuda:
                mask, targets = mask.cuda(), targets.cuda()
            class_weights = torch.ones(targets.shape).cuda()

            with nvtx.annotate(f"zero grad odd {n_iter/args.batch_size}", color="red"):
                model_1.zero_grad()
                model_3.zero_grad()
            
            with nvtx.annotate(f"model1 odd {n_iter/args.batch_size}", color="yellow"):
                bond_output = model_1(f_atoms.cuda(), f_bonds.cuda(), a2b.cuda(), b2a.cuda(), b2revb.cuda(), a_scope.cuda(), b_scope.cuda(), a2a.cuda(), features_batch)

            with nvtx.annotate(f"model3 odd {n_iter/args.batch_size}", color="orange"):
                atom_output = torch.zeros(f_atoms.size(0), args.hidden_size, requires_grad=False).cuda()
                preds = model_3(atom_output, bond_output, f_atoms, a2b, a_scope, features_batch)

            loss = loss_func(preds, targets) * class_weights * mask
            loss = loss.sum() / mask.sum()
        
            loss_sum += loss.item()
            iter_count += args.batch_size
            cum_loss_sum += loss.item()

            cum_iter_count += 1

            with nvtx.annotate(f"backward odd {n_iter/args.batch_size}", color="green"):
                loss.backward()

            with nvtx.annotate(f"optim odd {n_iter/args.batch_size}", color="blue"):
                optimizer_2.step()
                if isinstance(scheduler_2, NoamLR):
                    scheduler_2.step()
                optimizer_0.step()
                if isinstance(scheduler_0, NoamLR):
                    scheduler_0.step()

            n_iter += args.batch_size

                    #p.step()

    return n_iter, (cum_loss_sum / cum_iter_count)


def evaluate_rank1(model_1,
             model_3,
             data,
             loss_func,
             args,
             shared_dict,
             scaler = None,
             logger = None):

    model_1.eval()
    model_3.eval()
    preds = []
    loss_sum, iter_count = 0, 0
    val_mol_collator = MolCollator(args=args, shared_dict=shared_dict)
    val_mol_loader = DataLoader(data, batch_size=args.batch_size, shuffle=False, num_workers = 4,worker_init_fn=seed_worker,
                        generator=g, collate_fn=val_mol_collator)
    for _, item in enumerate(val_mol_loader):
        _, batch, features_batch2, mask, targets = item
        class_weights = torch.ones(targets.shape)
        targets = targets.cuda()
        mask = mask.cuda()
        class_weights = class_weights.cuda()

        with torch.no_grad():
            f_atoms, f_bonds, a2b, b2a, b2revb, a_scope, b_scope, a2a = batch
            
            bond_output = model_1(f_atoms.cuda(), f_bonds.cuda(), a2b.cuda(), b2a.cuda(), b2revb.cuda(), a_scope.cuda(), b_scope.cuda(), a2a.cuda(), features_batch2)
            atom_output = torch.zeros(f_atoms.size(0), args.hidden_size).cuda()
            batch_preds = model_3(atom_output, bond_output, f_atoms, a2b, a_scope, features_batch2)
 
            iter_count += 1
            if args.fingerprint:
                batch_preds.extend(batch_preds.data.cpu().numpy())
                continue

            if loss_func is not None:
                loss = loss_func(batch_preds, targets) * class_weights * mask
                loss = loss.sum() / mask.sum()
                loss_sum += loss.item()
            # Collect vectors
            batch_preds = batch_preds.data.cpu().numpy().tolist()
            if scaler is not None:
                batch_preds = scaler.inverse_transform(batch_preds)
            preds.extend(batch_preds)
    loss_avg = loss_sum / iter_count

    return loss_avg


def load_data(args, debug, logger):
    """
    load the training data.
    :param args:
    :param debug:
    :param logger:
    :return:
    """
    # Get data
    debug('Loading data')
    args.task_names = get_task_names(args.data_path)
    data = get_data(path=args.data_path, args=args, logger=logger)
    if data.data[0].features is not None:
        args.features_dim = len(data.data[0].features)
    else:
        args.features_dim = 0
    shared_dict = {}
    args.num_tasks = data.num_tasks()
    args.features_size = data.features_size()
    debug(f'Number of tasks = {args.num_tasks}')
    # Split data
    debug(f'Splitting data with seed {args.seed}')
    if args.separate_test_path:
        test_data = get_data(path=args.separate_test_path, args=args,
                             features_path=args.separate_test_features_path, logger=logger)
    if args.separate_val_path:
        val_data = get_data(path=args.separate_val_path, args=args,
                            features_path=args.separate_val_features_path, logger=logger)
    if args.separate_val_path and args.separate_test_path:
        train_data = data
    elif args.separate_val_path:
        train_data, _, test_data = split_data(data=data, split_type=args.split_type,
                                              sizes=(0.8, 0.2, 0.0), seed=args.seed, args=args, logger=logger)
    elif args.separate_test_path:
        train_data, val_data, _ = split_data(data=data, split_type=args.split_type,
                                             sizes=(0.8, 0.2, 0.0), seed=args.seed, args=args, logger=logger)
    else:
        train_data, val_data, test_data = split_data(data=data, split_type=args.split_type,
                                                     sizes=args.split_sizes, seed=args.seed, args=args, logger=logger)
    if args.dataset_type == 'classification':
        class_sizes = get_class_sizes(data)
        debug('Class sizes')
        for i, task_class_sizes in enumerate(class_sizes):
            debug(f'{args.task_names[i]} '
                  f'{", ".join(f"{cls}: {size * 100:.2f}%" for cls, size in enumerate(task_class_sizes))}')

    #if args.save_smiles_splits:
    #    save_splits(args, test_data, train_data, val_data)

    if args.features_scaling:
        features_scaler = train_data.normalize_features(replace_nan_token=0)
        val_data.normalize_features(features_scaler)
        test_data.normalize_features(features_scaler)
    else:
        features_scaler = None
    args.train_data_size = len(train_data)
    debug(f'Total size = {len(data):,} | '
          f'train size = {len(train_data):,} | val size = {len(val_data):,} | test size = {len(test_data):,}')

    # Initialize scaler and scale training targets by subtracting mean and dividing standard deviation (regression only)
    if args.dataset_type == 'regression':
        debug('Fitting scaler')
        _, train_targets = train_data.smiles(), train_data.targets()
        scaler = StandardScaler().fit(train_targets)
        scaled_targets = scaler.transform(train_targets).tolist()
        train_data.set_targets(scaled_targets)

        val_targets = val_data.targets()
        scaled_val_targets = scaler.transform(val_targets).tolist()
        val_data.set_targets(scaled_val_targets)
    else:
        scaler = None
    return features_scaler, scaler, shared_dict, test_data, train_data, val_data


def save_splits(args, test_data, train_data, val_data):
    """
    Save the splits.
    :param args:
    :param test_data:
    :param train_data:
    :param val_data:
    :return:
    """
    with open(args.data_path, 'r') as f:
        reader = csv.reader(f)
        header = next(reader)

        lines_by_smiles = {}
        indices_by_smiles = {}
        for i, line in enumerate(reader):
            smiles = line[0]
            lines_by_smiles[smiles] = line
            indices_by_smiles[smiles] = i

    all_split_indices = []
    for dataset, name in [(train_data, 'train'), (val_data, 'val'), (test_data, 'test')]:
        with open(os.path.join(args.save_dir, name + '_smiles.csv'), 'w') as f:
            writer = csv.writer(f)
            writer.writerow(['smiles'])
            for smiles in dataset.smiles():
                writer.writerow([smiles])
        with open(os.path.join(args.save_dir, name + '_full.csv'), 'w') as f:
            writer = csv.writer(f)
            writer.writerow(header)
            for smiles in dataset.smiles():
                writer.writerow(lines_by_smiles[smiles])
        split_indices = []
        for smiles in dataset.smiles():
            split_indices.append(indices_by_smiles[smiles])
            split_indices = sorted(split_indices)
        all_split_indices.append(split_indices)
    with open(os.path.join(args.save_dir, 'split_indices.pckl'), 'wb') as f:
        pickle.dump(all_split_indices, f)
    return writer


def train_pp(epoch, model, data, loss_func, optimizer, scheduler, 
            shared_dict, args, n_iter=0, logger=None, num_workers=4):

    model.train()

    loss_sum, iter_count = 0, 0
    cum_loss_sum, cum_iter_count = 0, 0

    mol_collator = MolCollator(shared_dict=shared_dict, args=args)

    #num_workers = 4
    if type(data) == DataLoader:
        mol_loader = data
    else:
        mol_loader = DataLoader(data, batch_size=args.batch_size, shuffle=False,
                            num_workers=num_workers, collate_fn=mol_collator, pin_memory=True)

    iteration=-1
    for _, item in enumerate(mol_loader):
        iteration+=1
        if iteration%args.num_micro_batch==0:
            micro_batches = []
            micro_batches.append(item)
            for i in range(1, args.num_micro_batch):
                micro_batches.append(next(enumerate(mol_loader))[1])

            with nvtx.annotate(f"rank {dist.get_rank()} step {n_iter/args.batch_size}"):
                step_time = time.time()

                with nvtx.annotate(f"zerograd even {n_iter/args.batch_size}", color="red"):
                    model.zero_grad()
                
                forward_only = False
                loss = forward_backward_step(model, micro_batches, args, forward_only)
                print("forward_backward_step done!")
                if args.node_rank==3:
                    loss_sum += loss.item()
                    cum_loss_sum += loss.item()
                iter_count += args.batch_size
                cum_iter_count += 1

                with nvtx.annotate(f"optim even {n_iter/args.batch_size}", color="blue"):
                    optimizer.step()
                    if isinstance(scheduler, NoamLR):
                        scheduler.step()

                n_iter += args.batch_size
        else:
            pass

    return n_iter, (cum_loss_sum / cum_iter_count)


def forward_backward_step(model, micro_batches, args, forward_only):

    loss_func = get_loss_func(args, model)
    #disable_grad_sync()
    num_micro_batch = args.num_micro_batch
    model_parallel_size = args.model_parallel_size
    num_warmup_microbatches = min(model_parallel_size - args.node_rank -1, num_micro_batch)
    num_microbatches_remaining = num_micro_batch - num_warmup_microbatches

    world_rank = dist.get_rank()

    input_tensors, output_tensors = [], []
    if not forward_only:
        input_batches, output_batches = [], []
    losses = []

    # Warmup forward passes
    # print(f"num_warmup_microbatches: {num_warmup_microbatches}")
    for i in range(num_warmup_microbatches):
        _, micro_batch = next(enumerate(micro_batches))
        _, batch, features_batch, mask, targets = micro_batch
        mask, targets = mask.cuda(), targets.cuda()
        class_weights = torch.ones(targets.shape).cuda()
        f_atoms_original, f_bonds_original, a2b, b2a, b2revb, a_scope, b_scope, a2a = batch
        # Recv forward
        if args.node_rank != 0:
            f_atoms_feature = torch.empty(f_atoms_original.size(0), args.hidden_size).cuda()
            f_bonds_feature = torch.empty(f_bonds_original.size(0), args.hidden_size).cuda()
            # print(f"recv fw: {f_atoms_feature.size()} at line 1974 to {world_rank-args.data_parallel_size}")
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
        output_batch = model(input_batch, features_batch)
        if args.node_rank==3:
            atom_output, bond_output = output_batch
        else:
            atom_output, bond_output, _, _, _, _, _, _, _, _ = output_batch
        # Send FW
        if args.node_rank != args.model_parallel_size - 1:
            # print(f"send fw: {atom_output.size()} at line 1993 to {world_rank+args.data_parallel_size}")
            dist.send(tensor=atom_output.cuda(), dst=world_rank + args.data_parallel_size)
            dist.send(tensor=bond_output.cuda(), dst=world_rank + args.data_parallel_size)
            

        if not forward_only:
            input_batches.append(input_batch)
            output_batches.append(output_batch)

    if num_microbatches_remaining > 0:
        # recv forward
        if args.node_rank != 0:
            if args.node_rank == args.model_parallel_size - 1:
                _, micro_batch = next(enumerate(micro_batches))
                _, batch, _, _, _ = micro_batch
                f_atoms_original, f_bonds_original,  _, _, _, _, _, _ = batch
            f_atoms_feature = torch.empty(f_atoms_original.size(0), args.hidden_size).cuda()
            f_bonds_feature = torch.empty(f_bonds_original.size(0), args.hidden_size).cuda()
            # print(f"recv fw: {f_atoms_feature.size()} at line 2014 from {world_rank-args.data_parallel_size}")
            dist.recv(tensor=f_atoms_feature, src=world_rank - args.data_parallel_size)
            dist.recv(tensor=f_bonds_feature, src=world_rank - args.data_parallel_size)
            
            f_atoms_feature.requires_grad = True
            f_bonds_feature.requires_grad = True
        else:
            f_atoms_feature = None
            f_bonds_feature = None

    # 1F1B
    # print("\nStart 1F1B")
    for i in range(num_microbatches_remaining):
        _, micro_batch = next(enumerate(micro_batches))
        _, batch, features_batch, mask, targets = micro_batch
        class_weights = torch.ones(targets.shape).cuda()
        mask, targets = mask.cuda(), targets.cuda()
        f_atoms_original, f_bonds_original, a2b, b2a, b2revb, a_scope, b_scope, a2a = batch
        input_batch = (f_atoms_original, f_bonds_original, a2b, b2a, b2revb, a_scope, b_scope, a2a, f_atoms_feature, f_bonds_feature)
        last_iteration = i == (num_microbatches_remaining - 1)
        # print(f"last_iteration: {last_iteration}")
        # Forward Step
        output_batch = model(input_batch, features_batch)
        if args.node_rank==3:
            atom_output, bond_output = output_batch
            pred = (atom_output + bond_output)/2
            loss = loss_func(pred, targets.cuda()) * class_weights * mask
            loss = loss.sum() / mask.sum()
            losses.append(loss)
        else:
            atom_output, bond_output, _, _, _, _, _, _, _, _ = output_batch

        if forward_only:
            # Send FW
            if args.node_rank != args.model_parallel_size - 1:
                dist.send(tensor=atom_output.cuda(), dst=world_rank+args.data_parallel_size)
                dist.send(tensor=bond_output.cuda(), dst=world_rank+args.data_parallel_size)
            if not last_iteration:
                # Recv forward
                if args.node_rank != 0:
                    f_atoms_feature = torch.empty(f_atoms_original.size(0), args.hidden_size).cuda()
                    f_bonds_feature = torch.empty(f_bonds_original.size(0), args.hidden_size).cuda()
                    f_atoms_feature.requires_grad = True
                    f_bonds_feature.requires_grad = True
                    dist.recv(tensor=f_atoms_feature.cuda(), src=world_rank-args.data_parallel_size)
                    dist.recv(tensor=f_bonds_feature.cuda(), src=world_rank-args.data_parallel_size)
        else:
            # Send FW recv BW
            if args.node_rank != args.model_parallel_size - 1:

                atom_output_grad, bond_output_grad = torch.zeros_like(atom_output), torch.zeros_like(bond_output)
                # Recv BW
                # print(f"recv bw: {atom_output_grad.size()} at line 2054 from {world_rank + args.data_parallel_size}")
                dist.recv(tensor=atom_output_grad.cuda(), src=world_rank + args.data_parallel_size)
                dist.recv(tensor=bond_output_grad.cuda(), src=world_rank + args.data_parallel_size)
                
                # Send FW
                # print(f"send fw: {atom_output.size()} at line 2052 to {world_rank + args.data_parallel_size}")
                dist.send(tensor=atom_output.cuda(), dst=world_rank + args.data_parallel_size)
                dist.send(tensor=bond_output.cuda(), dst=world_rank + args.data_parallel_size)

                output_batch = (f_atoms_original, f_bonds_original, a2b, b2a, b2revb, a_scope, b_scope,a2a, f_atoms_feature, f_bonds_feature)
                if f_atoms_feature is not None:
                    f_atoms_feature.retain_grad()
                    f_bonds_feature.retain_grad()
                input_batches.append(input_batch)
                output_batches.append(output_batch)

                if input_batches != []:
                    input_batch = input_batches.pop(0)
                    output_batch = output_batches.pop(0)

                # Backward Step
                atom_output.backward(atom_output_grad, retain_graph=True)
                bond_output.backward(bond_output_grad, retain_graph=True)
            else: # if last rank
                pred = (atom_output + bond_output)/2
                f_atoms_feature.retain_grad()
                f_bonds_feature.retain_grad()
                loss = loss_func(pred, targets.cuda()) * class_weights * mask
                loss = loss.sum() / mask.sum()
                losses.append(loss)
                loss.backward(retain_graph=True)

            if last_iteration:
                input_batch = []
                if args.node_rank != 0:
                    # send backward
                    # print(f"send bw: {f_atoms_feature.grad.size()} at line 2080 to {world_rank - args.data_parallel_size}")
                    dist.send(tensor=f_atoms_feature.grad, dst=world_rank - args.data_parallel_size)
                    dist.send(tensor=f_bonds_feature.grad, dst=world_rank - args.data_parallel_size)
                    
            else:
                # Send backward
                if args.node_rank != 0:
                    # print(f"send bw: {f_atoms_feature.grad.size()} at line 2085 to {world_rank - args.data_parallel_size}")
                    dist.send(tensor=f_atoms_feature.grad, dst=world_rank - args.data_parallel_size)
                    dist.send(tensor=f_bonds_feature.grad, dst=world_rank - args.data_parallel_size)
                    
                # Recv forward
                if args.node_rank != 0:
                    # print(f"recv fw: {f_atoms_feature.size()} at line 2089 from {world_rank - args.data_parallel_size}")
                    dist.recv(tensor=f_atoms_feature.cuda(), src=world_rank - args.data_parallel_size)
                    dist.recv(tensor=f_bonds_feature.cuda(), src=world_rank - args.data_parallel_size)
                    
    
    # cooldown phase
    # print("\nStart cooldown phase")
    if not forward_only:
        for i in range(num_warmup_microbatches):
            _, micro_batch = next(enumerate(micro_batches))
            _, batch, features_batch, mask, targets = micro_batch
            class_weights = torch.ones(targets.shape).cuda()
            mask, targets = mask.cuda(), targets.cuda()
            f_atoms, f_bonds, a2b, b2a, b2revb, a_scope, b_scope, a2a = batch
            f_atoms.requires_grad = True
            f_bonds.requires_grad = True
            if i == num_warmup_microbatches - 1:
                if args.node_rank == 0:
                    #enable_grad_sync()
                    pass
            atom_output_grad, bond_output_grad = torch.zeros_like(atom_output), torch.zeros_like(bond_output)
            if input_batches != []:
                input_batch = input_batches.pop(0)
                output_batch = output_batches.pop(0)
            
            if args.node_rank != args.model_parallel_size - 1:
                # recv backward
                # print(f"recv bw: {atom_output_grad.size()} at line 2113 from {world_rank+args.data_parallel_size}")
                dist.recv(tensor=atom_output_grad.cuda(), src=world_rank+args.data_parallel_size)
                dist.recv(tensor=bond_output_grad.cuda(), src=world_rank+args.data_parallel_size)
                torch.cuda.synchronize()
                
                # Backward Step
                atom_output.backward([atom_output_grad], retain_graph=True)
                bond_output.backward([bond_output_grad], retain_graph=True)
            else:
                atom_output, bond_output = output_batch
                pred = (atom_output + bond_output)/2
                loss = loss_func(pred, targets.cuda()) * class_weights * mask
                loss = loss.sum() / mask.sum()
                loss.backward()
            if args.node_rank != 0:
                # send backward
                # print(f"send bw: {atom_output.grad.size()} at line 2127 to {world_rank-args.data_parallel_size}")
                dist.send(tensor=atom_output.grad, dst=world_rank-args.data_parallel_size)
                dist.send(tensor=bond_output.grad, dst=world_rank-args.data_parallel_size)
                # release retained graph
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
                
    return loss
