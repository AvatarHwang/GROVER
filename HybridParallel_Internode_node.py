"""
Finetuning task with model parallelism usiong multi-processing.
"""


import torch
from torch import distributed as dist
import os
import time
import numpy as np
from torch.optim.lr_scheduler import ExponentialLR
import pandas as pd

#import modules
from torch.utils.data import DataLoader
from grover.data import MolCollator, StandardScaler

from ParallelModelVersion2 import Node_Block_parallel as model0
from ParallelModelVersion2 import NodeViewReadoutFFN as model2
from torch.nn.parallel import DistributedDataParallel as DDP

# Import modules of GROVER
from grover.util.utils import create_logger, get_loss_func, get_task_names, get_class_sizes, get_data, split_data, load_checkpoint, build_lr_scheduler, build_model, makedirs
from grover.util.parsing import parse_args
from grover.util.metrics import get_metric_func
from grover.util.scheduler import NoamLR
from task.predict import predict, evaluate_predictions
from grover.util.nn_utils import initialize_weights

import nvtx

# Controlling sources of randomness #
import random
torch.manual_seed(42)
torch.cuda.manual_seed_all(42)
np.random.seed(42)
random.seed(42)
torch.backends.cudnn.deterministic = True
g = torch.Generator()
g.manual_seed(42)   

#os.environ["NCCL_SOCKET_IFNAME"] = "eno1,ibp161s0f0"
#os.environ['NCCL_DEBUG']='INFO'

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

def init_process(world_rank, size, fn, backend='nccl'):
    """ Initialize the distributed environment. """
    dist.init_process_group(backend, rank=world_rank, world_size=size)
    fn(world_rank, size)

# Main training task
def run(rank, size):
    # Set cuda device for the rank
    rank = int(os.environ['LOCAL_RANK'])
    torch.cuda.set_device(rank)
    tag_id = int((rank)*100)

    # args
    if 1==1:
        args = parse_args()
        args.activation='PReLU'
        args.attn_hidden=4
        args.attn_out=128
        args.backbone='dualtrans'
        args.batch_size=32
        args.bias=False
        args.bond_drop_rate=0
        args.checkpoint_dir=None
        args.checkpoint_path='model/grover_large.pt'
        args.checkpoint_paths=['model/grover_large.pt']
        args.crossval_index_dir=None
        args.crossval_index_file=None
        args.cuda=True
        args.data_path='/home1/soonyear/GROVER/exampledata/finetune/tox21.csv'
        #args.data_path='exampledata/finetune/tox21.csv'
        args.dataset_type='classification'
        args.dense=False
        args.depth=6
        args.dist_coff=0.1
        args.distinct_init=False
        args.dropout=0.0
        args.early_stop_epoch=1000
        args.embedding_output_type='both'
        args.enbl_multi_gpu=False
        args.ensemble_size=1
        args.epochs=50
        args.features_dim=200
        args.features_generator=None
        args.features_only=False
        #args.features_path=['exampledata/finetune/tox21.npz']
        args.features_path=['exampledata/finetune/tox21.npz']
        args.features_scaling=False
        args.features_size=200
        args.ffn_hidden_size=700
        args.ffn_num_layers=2
        args.final_lr=0.0001
        args.fine_tune_coff=1
        args.fingerprint=False
        args.folds_file=None
        args.gpu=0
        args.hidden_size=1200
        args.init_lr=0.00015
        args.input_layer='fc'
        args.max_data_size=None
        args.max_lr=0.001
        args.metric='auc'
        args.minimize_score=False
        args.no_attach_fea=True
        args.no_cache=True
        args.num_attn_head=4
        args.num_folds=1
        args.num_lrs=1
        args.num_mt_block=1
        args.parser_name='finetune'
        args.save_dir='model/finetune/tox21/'
        args.save_smiles_splits=False
        args.seed=42
        args.select_by_loss=True
        args.self_attention=False
        args.separate_test_features_path=None
        args.separate_test_path=None
        args.separate_val_features_path=None
        args.separate_val_path=None
        args.show_individual_scores=False
        args.skip_epoch=0
        args.split_sizes=[0.8, 0.1, 0.1]
        args.split_type='scaffold_balanced'
        args.tensorboard=False
        args.test_fold_index=None
        args.undirected=False
        args.use_compound_names=False
        args.use_input_features=['exampledata/finetune/tox21.npz']
        args.val_fold_index=None
        args.warmup_epochs=2
        args.weight_decay=2e-07

    # Build logger
    logger = create_logger(name='train', save_dir=args.save_dir, quiet=False)
    debug, info = logger.debug, logger.info

    world_size = dist.get_world_size()
    group_node_ranks=[]
    group_edge_ranks=[]
    for i in range(world_size):
        if i<(world_size/2):
            group_node_ranks.append(i)
        else:
            group_edge_ranks.append(i)
    group_node = dist.new_group(ranks=group_node_ranks)
    group_edge = dist.new_group(ranks=group_edge_ranks)

    """cross_validate"""
    info = logger.info if logger is not None else print

    # Initialize relevant variables
    init_seed = args.seed
    save_dir = args.save_dir
    task_names = get_task_names(args.data_path)

    # Run training with different random seeds for each fold
    all_scores = []
    for fold_num in range(args.num_folds):
        info(f'Fold {fold_num}')
        args.seed = init_seed + fold_num
        args.save_dir = os.path.join(save_dir, f'fold_{fold_num}')
        makedirs(args.save_dir)
        if args.parser_name == "finetune":
            model_scores = run_training(args, logger, group_node, group_edge, tag_id)
        else:
            pass#model_scores = run_evaluation(args, logger)
        if rank==0:
            all_scores.append(model_scores)
    if rank==0:
        all_scores = np.array(all_scores)

        # Report scores for each fold
        info(f'{args.num_folds}-fold cross validation')

        for fold_num, scores in enumerate(all_scores):
            info(f'Seed {init_seed + fold_num} ==> test {args.metric} = {np.nanmean(scores):.6f}')

            if args.show_individual_scores:
                for task_name, score in zip(task_names, scores):
                    info(f'Seed {init_seed + fold_num} ==> test {task_name} {args.metric} = {score:.6f}')

        # Report scores across models
        avg_scores = np.nanmean(all_scores, axis=1)  # average score for each model across tasks
        mean_score, std_score = np.nanmean(avg_scores), np.nanstd(avg_scores)
        info(f'overall_{args.split_type}_test_{args.metric}={mean_score:.6f}')
        info(f'std={std_score:.6f}')

        if args.show_individual_scores:
            for task_num, task_name in enumerate(task_names):
                info(f'Overall test {task_name} {args.metric} = '
                    f'{np.nanmean(all_scores[:, task_num]):.6f} +/- {np.nanstd(all_scores[:, task_num]):.6f}')

"""Run training for a fold"""
def run_training(args, logger, group_node, group_edge, tag_id):
    if logger is not None:
        debug, info = logger.debug, logger.info
    else:
        debug = info = print

    # Load data and molvocab
    features_scaler, scaler, shared_dict, test_data, train_data, val_data = load_data(args, debug, logger)

    # Metric Function
    metric_func = get_metric_func(metric=args.metric)

    # Set up test set evaluation
    test_smiles, test_targets = test_data.smiles(), test_data.targets()
    sum_test_preds = np.zeros((len(test_smiles), args.num_tasks))


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

    # Get loss function
    loss_func = get_loss_func(args, model)

    world_size = dist.get_world_size()

    num_workers=4*torch.cuda.device_count()#int(torch.multiprocessing.cpu_count() / (world_size/2))
    print("num_workers", num_workers)

    # Process 0
    #get world rank
    world_rank = dist.get_rank()
    rank = int(os.environ['RANK'])
    if world_rank<dist.get_world_size()/2:
        training_time = []
        loss_measure = []

        # Build models and optimizers
        model_0 = model0(model=model, rank=rank).cuda()
        model_2 = model2(model=model, rank=rank, args=args).cuda()
        
        if world_size > 2:
            model_0 = DDP(model_0, process_group=group_node)
            model_2 = DDP(model_2, process_group=group_node, find_unused_parameters=True)

        optimizer_0 = build_optimizer(model_0, args)
        optimizer_2 = build_optimizer(model_2, args)

        # Ensure that model is saved in correct location for evaluation if 0 epochs
        if rank==0:
            torch.save(model_0.state_dict(), os.path.join(save_dir, "model0.pt"))
            torch.save(model_2.state_dict(), os.path.join(save_dir, "model2.pt"))
                    
        # Build learning rate scheduler
        scheduler_0 = build_lr_scheduler(optimizer_0, args)
        scheduler_2 = build_lr_scheduler(optimizer_2, args)

        # Set up DataLoader
        mol_collator = MolCollator(shared_dict={}, args=args)
        
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
                                    num_workers=10,
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
            n_iter, train_loss = train(
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
            val_scores, val_loss = evaluate(model_0=model_0,
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
            if isinstance(scheduler_0, ExponentialLR):
                scheduler_0.step()
            if isinstance(scheduler_2, ExponentialLR):
                scheduler_2.step()

            if args.show_individual_scores:
                # Individual validation scores
                for task_name, val_score in zip(args.task_names, val_scores):
                    debug(f'Validation {task_name} {args.metric} = {val_score:.6f}')
            loss_measure.append(val_loss)
            if rank==0:
                print('Epoch: {:04d}'.format(epoch),
                    'loss_train: {:.6f}'.format(train_loss),
                    'loss_val: {:.6f}'.format(val_loss),
                    f'{args.metric}_val: {avg_val_score:.4f}',
                    # 'auc_val: {:.4f}'.format(avg_val_score),
                    'cur_lr_0: {:.5f}'.format(scheduler_0.get_lr()[-1]),
                    'cur_lr_2: {:.5f}'.format(scheduler_2.get_lr()[-1]),
                    't_time: {:.4f}s'.format(t_time),
                    'v_time: {:.4f}s'.format(v_time))

            # Save model checkpoint if improved validation score
            if rank==0:
                if args.select_by_loss:
                    if val_loss < min_val_loss:
                        min_val_loss, best_epoch = val_loss, epoch
                        torch.save(model_0.state_dict(), os.path.join(save_dir, "model0.pt"))
                        torch.save(model_2.state_dict(), os.path.join(save_dir, "model2.pt"))
                else:
                    if args.minimize_score and avg_val_score < best_score or \
                            not args.minimize_score and avg_val_score > best_score:
                        best_score, best_epoch = avg_val_score, epoch
                        torch.save(model_0.state_dict(), os.path.join(save_dir, "model0.pt"))
                        torch.save(model_2.state_dict(), os.path.join(save_dir, "model2.pt"))
    
            if epoch - best_epoch > args.early_stop_epoch:
                break
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
            
            test_preds, _ = predict(
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
            print(f"\nloss_measure time is:{loss_measure}")

            return ensemble_scores
    else:
        pass
            

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

def train(epoch, model_0, model_2, data, loss_func, optimizer_0, optimizer_2, scheduler_0,
          scheduler_2, shared_dict, args, n_iter = 0,
          logger = None, num_workers=4):

    rank = int(os.environ['RANK'])

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


def evaluate(model_0,
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

def predict(model_0,
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

def get_ffn_layer_id(model):
    return [id(x) for x in model.state_dict() if "mol" in x and "ffn" in x]


def build_optimizer(model, args):
    # Only adjust the learning rate for the GroverFinetuneTask.
    if args.parser_name=='finetune':
        ffn_params = get_ffn_layer_id(model)
    else:
        # if not, init adam optimizer normally.
        return torch.optim.Adam(model.parameters(), lr=args.init_lr, weight_decay=args.weight_decay)
    base_params = filter(lambda p: id(p) not in ffn_params, model.parameters())
    ffn_params = filter(lambda p: id(p) in ffn_params, model.parameters())
    if args.fine_tune_coff == 0:
        for param in base_params:
            param.requires_grad = False

    optimizer = torch.optim.Adam([
        {'params': base_params, 'lr': args.init_lr * args.fine_tune_coff},
        {'params': ffn_params, 'lr': args.init_lr}
    ], lr=args.init_lr, weight_decay=args.weight_decay)

    return optimizer


if __name__ == "__main__":
    # Initialize Rank and Distributed Learning Environment
    world_size = 8
    world_rank = int(os.environ["RANK"])
    init_process(world_rank, world_size, run)
