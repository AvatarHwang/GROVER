"""
The GROVER trainer.
"""
import os
import time
from logging import Logger
from typing import List, Tuple
from collections.abc import Callable
import torch
from torch.nn import Module
from torch.utils.data import DataLoader

from grover.model.models import GroverTask, GroverTask_for_pp
from grover.util.multi_gpu_wrapper import MultiGpuWrapper as mgw

from task.dapple import forward_backward_step

import copy

from torch import Tensor
from typing import Callable, Iterable, Iterator, List, Tuple, Union, cast
Tensors = Tuple[Tensor, ...]
TensorOrTensors = Union[Tensor, Tensors]
Function = Callable[[TensorOrTensors], TensorOrTensors]


class GROVERTrainer:
    def __init__(self,
                 args,
                 embedding_model: Module,
                 atom_vocab_size: int,  # atom vocab size
                 bond_vocab_size: int,
                 fg_szie: int,
                 train_dataloader: DataLoader,
                 test_dataloader: DataLoader,
                 optimizer_builder: Callable,
                 scheduler_builder: Callable,
                 logger: Logger = None,
                 with_cuda: bool = False,
                 enable_multi_gpu: bool = False):
        """
        The init function of GROVERTrainer
        :param args: the input arguments.
        :param embedding_model: the model to generate atom/bond embeddings.
        :param atom_vocab_size: the vocabulary size of atoms.
        :param bond_vocab_size: the vocabulary size of bonds.
        :param fg_szie: the size of semantic motifs (functional groups)
        :param train_dataloader: the data loader of train data.
        :param test_dataloader: the data loader of validation data.
        :param optimizer_builder: the function of building the optimizer.
        :param scheduler_builder: the function of building the scheduler.
        :param logger: the logger
        :param with_cuda: enable gpu training.
        :param enable_multi_gpu: enable multi_gpu traning.
        """

        self.args = args
        self.with_cuda = with_cuda
        self.grover = embedding_model
        if args.pipeline_parallel:
            self.model = GroverTask_for_pp(args, embedding_model, atom_vocab_size, bond_vocab_size, fg_szie)
        else:
            self.model = GroverTask(args, embedding_model, atom_vocab_size, bond_vocab_size, fg_szie)
        self.loss_func = self.model.get_loss_func(args)
        self.enable_multi_gpu = enable_multi_gpu

        self.atom_vocab_size = atom_vocab_size
        self.bond_vocab_size = bond_vocab_size
        self.debug = logger.debug if logger is not None else print

        if self.with_cuda:
            # print("Using %d GPUs for training." % (torch.cuda.device_count()))
            self.model = self.model.cuda()

        self.train_data = train_dataloader
        self.test_data = test_dataloader

        self.optimizer = optimizer_builder(self.model, self.args)
        self.scheduler = scheduler_builder(self.optimizer, self.args)
        if self.enable_multi_gpu:
            self.optimizer = mgw.DistributedOptimizer(self.optimizer,
                                                      named_parameters=self.model.named_parameters())
        self.args = args
        self.n_iter = 0

        self.pipeline_parallel = args.pipeline_parallel
        self.micro_batch_size = args.micro_batch_size

    def broadcast_parameters(self) -> None:
        """
        Broadcast parameters before training.
        :return: no return.
        """
        if self.enable_multi_gpu:
            # broadcast parameters & optimizer state.
            mgw.broadcast_parameters(self.model.state_dict(), root_rank=0)
            mgw.broadcast_optimizer_state(self.optimizer, root_rank=0)

    def train(self, epoch: int) -> List:
        """
        The training iteration
        :param epoch: the current epoch number.
        :return: the loss terms of current epoch.
        """
        # return self.mock_iter(epoch, self.train_data, train=True)
        return self.iter(epoch, self.train_data, train=True)

    def test(self, epoch: int) -> List:
        """
        The test/validaiion iteration
        :param epoch: the current epoch number.
        :return:  the loss terms as a list
        """
        # return self.mock_iter(epoch, self.test_data, train=False)
        return self.iter(epoch, self.test_data, train=False)

    def mock_iter(self, epoch: int, data_loader: DataLoader, train: bool = True) -> List:
        """
        Perform a mock iteration. For test only.
        :param epoch: the current epoch number.
        :param data_loader: the data loader.
        :param train: True: train model, False: validation model.
        :return: the loss terms as a list
        """

        for _, _ in enumerate(data_loader):
            self.scheduler.step()
        cum_loss_sum = 0.0
        self.n_iter += self.args.batch_size
        return self.n_iter, cum_loss_sum, (0, 0, 0, 0, 0, 0)

    def iter(self, epoch, data_loader, train=True) -> List:
        """
        Perform a training / validation iteration.
        :param epoch: the current epoch number.
        :param data_loader: the data loader.
        :param train: True: train model, False: validation model.
        :return: the loss terms as a list
        """
        data_loader_cp = copy.deepcopy(data_loader)
        if train:
            self.model.train()
        else:
            self.model.eval()

        loss_sum, iter_count = 0, 0
        cum_loss_sum, cum_iter_count = 0, 0
        av_loss_sum, bv_loss_sum, fg_loss_sum, av_dist_loss_sum, bv_dist_loss_sum, fg_dist_loss_sum = 0, 0, 0, 0, 0, 0
        loss_func = self.model.get_loss_func(self.args)

        if self.pipeline_parallel:
            with torch.profiler.profile(
                activities=[
                    torch.profiler.ProfilerActivity.CPU,
                    torch.profiler.ProfilerActivity.CUDA,
                ],
                schedule=torch.profiler.schedule(
                    wait=0,
                    warmup=0,
                    active=2,
                    repeat=1),
                on_trace_ready=torch.profiler.tensorboard_trace_handler('./log'),
                with_stack=True,
                with_flops=True,
                with_modules=True
                ) as p:
                    #while(len(data_loader_cp.dataset.data)>=self.args.num_micro_batch):
                    while(len(data_loader_cp.dataset.data)>0):
                        mini_batch = next(iter(data_loader_cp))
                        # split item into micro batches
                        f_atoms_original, f_bonds_original, a2b, b2a, b2revb, a_scope, b_scope, a2a = mini_batch["graph_input"]
                        f_atoms_original = split_minibatch(f_atoms_original, self.args.num_micro_batch)
                        f_bonds_original = split_minibatch(f_bonds_original, self.args.num_micro_batch)
                        a2b = split_minibatch(a2b, self.args.num_micro_batch)
                        b2a = split_minibatch(b2a, self.args.num_micro_batch)
                        b2revb = split_minibatch(b2revb, self.args.num_micro_batch)
                        a_scope = split_minibatch(a_scope, self.args.num_micro_batch)
                        b_scope = split_minibatch(b_scope, self.args.num_micro_batch)
                        a2a = split_minibatch(a2a, self.args.num_micro_batch)
                        targets_av_task = split_minibatch(mini_batch["targets"]["av_task"], self.args.num_micro_batch)
                        targets_bv_task = split_minibatch(mini_batch["targets"]["bv_task"], self.args.num_micro_batch)
                        targets_fg_task = split_minibatch(mini_batch["targets"]["fg_task"], self.args.num_micro_batch)
                        micro_batches = []
                        for i in range(self.args.num_micro_batch):
                            micro_batch = {}
                            micro_batch["graph_input"] = (f_atoms_original[i], f_bonds_original[i], a2b[i], b2a[i], b2revb[i], a_scope[i], b_scope[i], a2a[i])
                            micro_batch["targets"] = {}
                            micro_batch["targets"]["av_task"] = targets_av_task[i]
                            micro_batch["targets"]["bv_task"] = targets_bv_task[i]
                            micro_batch["targets"]["fg_task"] = targets_fg_task[i]
                            micro_batches.append(micro_batch)
                        
                        # for i in range(self.args.num_micro_batch):
                        #     micro_batches.append(next(iter(data_loader_cp)))
                        #     data_loader_cp.dataset.data = data_loader_cp.dataset.data[1:]

                        if len(micro_batches)<self.args.num_micro_batch:
                            break
                        
                        forward_only = False if train else True
                        losses = forward_backward_step(self.model, micro_batches, self.args, forward_only, loss_func)
                        if self.args.node_rank == 3:
                            if losses != []:
                                for losse in losses:
                                    loss, av_loss, bv_loss, fg_loss, av_dist_loss, bv_dist_loss, fg_dist_loss = losse
                                    loss_sum += loss.item()
                                    av_loss_sum += av_loss.item()
                                    bv_loss_sum += bv_loss.item()
                                    fg_loss_sum += fg_loss.item()
                                    av_dist_loss_sum += av_dist_loss.item() if type(av_dist_loss) != float else av_dist_loss
                                    bv_dist_loss_sum += bv_dist_loss.item() if type(bv_dist_loss) != float else bv_dist_loss
                                    fg_dist_loss_sum += fg_dist_loss.item() if type(fg_dist_loss) != float else fg_dist_loss
                            iter_count += self.args.batch_size

                            if train:
                                cum_loss_sum += loss.item()
                                # Run model
                                # self.model.zero_grad()
                                self.optimizer.zero_grad()
                                # loss.backward()
                                self.optimizer.step()
                                self.scheduler.step()
                            else:
                                # For eval model, only consider the loss of three task.
                                cum_loss_sum += av_loss.item()
                                cum_loss_sum += bv_loss.item()
                                cum_loss_sum += fg_loss.item()

                        cum_iter_count += 1
                        self.n_iter += self.args.batch_size

                        p.step()

            if self.args.node_rank==3:
                if cum_iter_count != 0:
                    cum_loss_sum /= cum_iter_count
                    av_loss_sum /= cum_iter_count
                    bv_loss_sum /= cum_iter_count
                    fg_loss_sum /= cum_iter_count
                    av_dist_loss_sum /= cum_iter_count
                    bv_dist_loss_sum /= cum_iter_count
                    fg_dist_loss_sum /= cum_iter_count
            else:
                return self.n_iter, None, (None, None, None, None, None, None)

            return self.n_iter, cum_loss_sum, (av_loss_sum, bv_loss_sum, fg_loss_sum, av_dist_loss_sum,
                                            bv_dist_loss_sum, fg_dist_loss_sum)
        else:
            for _, item in enumerate(data_loader):
                batch_graph = item["graph_input"]
                targets = item["targets"]

                if next(self.model.parameters()).is_cuda:
                    targets["av_task"] = targets["av_task"].cuda()
                    targets["bv_task"] = targets["bv_task"].cuda()
                    targets["fg_task"] = targets["fg_task"].cuda()

                preds = self.model(batch_graph)

                # # ad-hoc code, for visualizing a model, comment this block when it is not needed
                # import dglt.contrib.grover.vis_model as vis_model
                # for task in ['av_task', 'bv_task', 'fg_task']:
                #     vis_graph = vis_model.make_dot(self.model(batch_graph)[task],
                #                                    params=dict(self.model.named_parameters()))
                #     # vis_graph.view()
                #     vis_graph.render(f"{self.args.backbone}_model_{task}_vis.png", format="png")
                # exit()

                loss, av_loss, bv_loss, fg_loss, av_dist_loss, bv_dist_loss, fg_dist_loss = self.loss_func(preds, targets)

                loss_sum += loss.item()
                iter_count += self.args.batch_size

                if train:
                    cum_loss_sum += loss.item()
                    # Run model
                    self.model.zero_grad()
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()
                    self.scheduler.step()
                else:
                    # For eval model, only consider the loss of three task.
                    cum_loss_sum += av_loss.item()
                    cum_loss_sum += bv_loss.item()
                    cum_loss_sum += fg_loss.item()

                av_loss_sum += av_loss.item()
                bv_loss_sum += bv_loss.item()
                fg_loss_sum += fg_loss.item()
                av_dist_loss_sum += av_dist_loss.item() if type(av_dist_loss) != float else av_dist_loss
                bv_dist_loss_sum += bv_dist_loss.item() if type(bv_dist_loss) != float else bv_dist_loss
                fg_dist_loss_sum += fg_dist_loss.item() if type(fg_dist_loss) != float else fg_dist_loss

                cum_iter_count += 1
                self.n_iter += self.args.batch_size

                # Debug only.
                # if i % 50 == 0:
                #     print(f"epoch: {epoch}, batch_id: {i}, av_loss: {av_loss}, bv_loss: {bv_loss}, "
                #           f"fg_loss: {fg_loss}, av_dist_loss: {av_dist_loss}, bv_dist_loss: {bv_dist_loss}, "
                #           f"fg_dist_loss: {fg_dist_loss}")

            cum_loss_sum /= cum_iter_count
            av_loss_sum /= cum_iter_count
            bv_loss_sum /= cum_iter_count
            fg_loss_sum /= cum_iter_count
            av_dist_loss_sum /= cum_iter_count
            bv_dist_loss_sum /= cum_iter_count
            fg_dist_loss_sum /= cum_iter_count

            return self.n_iter, cum_loss_sum, (av_loss_sum, bv_loss_sum, fg_loss_sum, av_dist_loss_sum,
                                            bv_dist_loss_sum, fg_dist_loss_sum)

    def save(self, epoch, file_path, name=None) -> str:
        """
        Save the intermediate models during training.
        :param epoch: the epoch number.
        :param file_path: the file_path to save the model.
        :return: the output path.
        """
        # add specific time in model fine name, in order to distinguish different saved models
        now = time.localtime()
        if name is None:
            name = "_%04d_%02d_%02d_%02d_%02d_%02d" % (
                now.tm_year, now.tm_mon, now.tm_mday, now.tm_hour, now.tm_min, now.tm_sec)
        output_path = file_path + name + ".ep%d" % epoch
        scaler = None
        features_scaler = None
        state = {
            'args': self.args,
            'state_dict': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'scheduler_step': self.scheduler.current_step,
            "epoch": epoch,
            'data_scaler': {
                'means': scaler.means,
                'stds': scaler.stds
            } if scaler is not None else None,
            'features_scaler': {
                'means': features_scaler.means,
                'stds': features_scaler.stds
            } if features_scaler is not None else None
        }
        torch.save(state, output_path)

        # Is this necessary?
        # if self.with_cuda:
        #    self.model = self.model.cuda()
        print("EP:%d Model Saved on:" % epoch, output_path)
        return output_path

    def save_tmp(self, epoch, file_path, rank=0):
        """
        Save the models for auto-restore during training.
        The model are stored in file_path/tmp folder and will replaced on each epoch.
        :param epoch: the epoch number.
        :param file_path: the file_path to store the model.
        :param rank: the current rank (decrypted).
        :return:
        """
        store_path = os.path.join(file_path, "tmp")
        if not os.path.exists(store_path):
            os.makedirs(store_path, exist_ok=True)
        store_path = os.path.join(store_path, "model.%d" % rank)
        state = {
            'args': self.args,
            'state_dict': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'scheduler_step': self.scheduler.current_step,
            "epoch": epoch
        }
        torch.save(state, store_path)

    def restore(self, file_path, rank=0) -> Tuple[int, int]:
        """
        Restore the training state saved by save_tmp.
        :param file_path: the file_path to store the model.
        :param rank: the current rank (decrypted).
        :return: the restored epoch number and the scheduler_step in scheduler.
        """
        cpt_path = os.path.join(file_path, "tmp", "model.%d" % rank)
        if not os.path.exists(cpt_path):
            print("No checkpoint found %d")
            return 0, 0
        cpt = torch.load(cpt_path)
        self.model.load_state_dict(cpt["state_dict"])
        self.optimizer.load_state_dict(cpt["optimizer"])
        epoch = cpt["epoch"]
        scheduler_step = cpt["scheduler_step"]
        self.scheduler.current_step = scheduler_step
        print("Restore checkpoint, current epoch: %d" % (epoch))
        return epoch, scheduler_step


def split_minibatch(input: TensorOrTensors, chunks: int):
    """Splits an input mini-batch into multiple micro-batches."""
    inputs: Iterable[TensorOrTensors]

    if isinstance(input, Tensor):
        inputs = input.chunk(chunks)
    else:
        rotated: List[Tensors] = []

        for tensor in input:
            tensors = tensor.chunk(chunks)
            rotated.append(cast(Tensors, tensors))

        inputs = zip(*rotated)

    return [x for x in inputs]