'''
Models for Model Parallel
'''

import torch
from torch import nn
# from grover.util.utils import load_checkpoint
from grover.model.layers import Readout
import numpy as np
from typing import List, Dict, Callable, Union

from grover.util.nn_utils import get_activation_function, select_neighbor_and_aggregate

import copy
from torch import distributed as dist

import time
import os

from argparse import Namespace

from grover.data import get_atom_fdim, get_bond_fdim

from grover.model.layers import MTBlock, PositionwiseFeedForward, SublayerConnection

class Node_Block_parallel(nn.Module):
    def __init__(self, model, rank):
        super(Node_Block_parallel, self).__init__()

        self.rank = rank
        self.node_blocks = copy.deepcopy(model.grover.encoders.node_blocks.cuda())

    def forward(self, f_atoms, f_bonds, a2b, b2a, b2revb, a_scope, b_scope, a2a, features_batch):
        node_batch = f_atoms, f_bonds, a2b, b2a, b2revb, a_scope, b_scope, a2a
        original_f_atoms, original_f_bonds = f_atoms, f_bonds
        
        for nb in self.node_blocks:
            node_batch, features_batch = nb(node_batch, features_batch)

        atom_output, _, _, _, _, _, _, _ = node_batch

        return atom_output.cuda()


class NodeViewReadoutFFN(nn.Module):
    def __init__(self, model, rank, args):
        super(NodeViewReadoutFFN, self).__init__()
        self.hidden_size = args.hidden_size
        self.num_tasks = args.num_tasks
        self.embedding_output_type = args.embedding_output_type
        self.ffn_atom_from_atom = copy.deepcopy(model.grover.encoders.ffn_atom_from_atom.cuda())
        self.atom_from_atom_sublayer = copy.deepcopy(model.grover.encoders.atom_from_atom_sublayer.cuda())
        self.ffn_atom_from_bond = copy.deepcopy(model.grover.encoders.ffn_atom_from_bond.cuda())

        self.atom_from_bond_sublayer = copy.deepcopy(model.grover.encoders.atom_from_bond_sublayer.cuda())

        self.act_func_node = copy.deepcopy(model.grover.encoders.act_func_node.cuda())

        self.readout = copy.deepcopy(model.readout.cuda())

        self.mol_atom_from_atom_ffn = copy.deepcopy(model.mol_atom_from_atom_ffn.cuda())

        self.classification = args.dataset_type == 'classification'
        if self.classification:
            self.sigmoid = nn.Sigmoid()

    def atom_bond_transform(self,
                            atomwise_input=None,
                            original_f_atoms=None,
                            a2a=None,
                            rank=0,
                            tag_id=0
                            ):
        # atom input to atom output
        atomwise_input, _ = self.pointwise_feed_forward_to_atom_embedding(atomwise_input, original_f_atoms, a2a,
                                                                        self.ffn_atom_from_atom)
        atom_in_atom_out = self.atom_from_atom_sublayer(None, atomwise_input)

        dist.isend(atom_in_atom_out, rank+int(dist.get_world_size()/2), None)

        return atom_in_atom_out

    def create_ffn(self, args):
        # Note: args.features_dim is set according the real loaded features data
        if args.features_only:
            first_linear_dim = args.features_size + args.features_dim
        else:
            if args.self_attention:
                first_linear_dim = args.hidden_size * args.attn_out
                # TODO: Ad-hoc!
                # if args.use_input_features:
                first_linear_dim += args.features_dim
            else:
                first_linear_dim = args.hidden_size + args.features_dim

        dropout = nn.Dropout(args.dropout)
        activation = get_activation_function(args.activation)
        # TODO: ffn_hidden_size
        # Create FFN layers
        if args.ffn_num_layers == 1:
            ffn = [
                dropout.cuda(),
                nn.Linear(first_linear_dim, args.output_size).cuda()
            ]
        else:
            ffn = [
                dropout.cuda(),
                nn.Linear(first_linear_dim, args.ffn_hidden_size).cuda()
            ]
            for _ in range(args.ffn_num_layers - 2):
                ffn.extend([
                    activation.cuda(),
                    dropout.cuda(),
                    nn.Linear(args.ffn_hidden_size, args.ffn_hidden_size).cuda(),
                ])
            ffn.extend([
                activation.cuda(),
                dropout.cuda(),
                nn.Linear(args.ffn_hidden_size, args.output_size).cuda(),
            ])

        # Create FFN model
        return nn.Sequential(*ffn)

    def pointwise_feed_forward_to_atom_embedding(self, emb_output, atom_fea, index, ffn_layer):
        aggr_output = select_neighbor_and_aggregate(emb_output, index)
        aggr_outputx = torch.cat([atom_fea.cuda(), aggr_output.cuda()], dim=1)
        return ffn_layer(aggr_outputx), aggr_output

    def forward(self, atom_output, original_f_atoms, a2a, a_scope, features_batch):
        rank = dist.get_rank()
        local_rank = int(os.environ['LOCAL_RANK'])
        #tag_id = int((local_rank)*100 + rank*1000)
        tag_id = local_rank * 100
        atom_embeddings = self.atom_bond_transform(
                                        atomwise_input=atom_output.cuda(),
                                        original_f_atoms=original_f_atoms,
                                        a2a=a2a,
                                        rank=rank,
                                        tag_id=tag_id)
        atom_in_atom_out = atom_embeddings

        a_scope = a_scope.data.cpu().numpy().tolist()
        mol_atom_from_atom_output = self.readout(atom_in_atom_out, a_scope)
        if features_batch[0] is not None:
            features_batch = torch.from_numpy(np.stack(features_batch)).float()
            if 1: # if cuda
                features_batch = features_batch.cuda()
            features_batch = features_batch.to(atom_in_atom_out)
            if len(features_batch.shape) == 1:
                features_batch = features_batch.view([1, features_batch.shape[0]])
        else:
            features_batch = None

        if features_batch is not None:
            mol_atom_from_atom_output = torch.cat([mol_atom_from_atom_output, features_batch], 1)
        if self.training:
            atom_ffn_output = self.mol_atom_from_atom_ffn(mol_atom_from_atom_output)

            # Recv bond_ffn_output
            bond_ffn_output = torch.zeros(atom_ffn_output.size(0), self.num_tasks).cuda()
            dist.recv(bond_ffn_output, rank+int(dist.get_world_size()/2), None)
            # Send atom_ffn_output
            dist.isend(atom_ffn_output, rank+int(dist.get_world_size()/2), None)

            return atom_ffn_output, bond_ffn_output.cuda()
        else:
            atom_ffn_output = self.mol_atom_from_atom_ffn(mol_atom_from_atom_output)
            if self.classification:
                atom_ffn_output = self.sigmoid(atom_ffn_output)

            # Recv bond_ffn_output
            bond_ffn_output = torch.zeros(atom_ffn_output.size(0), self.num_tasks).cuda()
            dist.recv(bond_ffn_output, rank+int(dist.get_world_size()/2), None)
            # Send atom_ffn_output
            dist.isend(atom_ffn_output, rank+int(dist.get_world_size()/2), None)

            output = (atom_ffn_output + bond_ffn_output.cuda()) / 2

        return output


class Edge_Block_parallel(nn.Module):
    def __init__(self, model, rank):
        super(Edge_Block_parallel, self).__init__()
        self.rank = rank
        self.edge_blocks = copy.deepcopy(model.grover.encoders.edge_blocks.cuda())

    def forward(self, f_atoms, f_bonds, a2b, b2a, b2revb, a_scope, b_scope, a2a, features_batch):
        edge_batch = f_atoms, f_bonds, a2b, b2a, b2revb, a_scope, b_scope, a2a

        for eb in self.edge_blocks:
            edge_batch, features_batch = eb(edge_batch, features_batch)

        _, bond_output, _, _, _, _, _, _  = edge_batch

        return bond_output.cuda()

class EdgeViewReadoutFFN(nn.Module):
    def __init__(self, model, rank, args):
        super(EdgeViewReadoutFFN, self).__init__()
        self.hidden_size = args.hidden_size
        self.num_tasks = args.num_tasks
        self.rank = rank
        self.embedding_output_type = args.embedding_output_type

        self.ffn_atom_from_atom = copy.deepcopy(model.grover.encoders.ffn_atom_from_atom.cuda())
        self.ffn_atom_from_bond = copy.deepcopy(model.grover.encoders.ffn_atom_from_bond.cuda())
        self.atom_from_atom_sublayer = copy.deepcopy(model.grover.encoders.atom_from_atom_sublayer.cuda())
        self.atom_from_bond_sublayer = copy.deepcopy(model.grover.encoders.atom_from_bond_sublayer.cuda())

        self.act_func_edge = copy.deepcopy(model.grover.encoders.act_func_edge.cuda())

        self.readout = copy.deepcopy(model.readout.cuda())

        self.mol_atom_from_bond_ffn = copy.deepcopy(model.mol_atom_from_bond_ffn.cuda())
        
        self.classification = args.dataset_type == 'classification'
        if self.classification:
            self.sigmoid = nn.Sigmoid()

    def atom_bond_transform(self,
                            atomwise_input=None,
                            bondwise_input=None,
                            original_f_atoms=None,
                            a2b=None,
                            rank=0,
                            tag_id=0
                            ):
        # bond to atom
        bondwise_input, _ = self.pointwise_feed_forward_to_atom_embedding(bondwise_input, original_f_atoms, a2b,
                                                                        self.ffn_atom_from_bond)
        bond_in_atom_out = self.atom_from_bond_sublayer(None, bondwise_input)

        dist.recv(atomwise_input, rank-int(dist.get_world_size()/2), None)
        atomwise_input.detach()

        return atomwise_input, bond_in_atom_out

    def create_ffn(self, args):
        # Note: args.features_dim is set according the real loaded features data
        if args.features_only:
            first_linear_dim = args.features_size + args.features_dim
        else:
            if args.self_attention:
                first_linear_dim = args.hidden_size * args.attn_out
                # TODO: Ad-hoc!
                # if args.use_input_features:
                first_linear_dim += args.features_dim
            else:
                first_linear_dim = args.hidden_size + args.features_dim

        dropout = nn.Dropout(args.dropout)
        activation = get_activation_function(args.activation)
        # TODO: ffn_hidden_size
        # Create FFN layers
        if args.ffn_num_layers == 1:
            ffn = [
                dropout.cuda(),
                nn.Linear(first_linear_dim, args.output_size).cuda()
            ]
        else:
            ffn = [
                dropout.cuda(),
                nn.Linear(first_linear_dim, args.ffn_hidden_size).cuda()
            ]
            for _ in range(args.ffn_num_layers - 2):
                ffn.extend([
                    activation.cuda(),
                    dropout.cuda(),
                    nn.Linear(args.ffn_hidden_size, args.ffn_hidden_size).cuda(),
                ])
            ffn.extend([
                activation.cuda(),
                dropout.cuda(),
                nn.Linear(args.ffn_hidden_size, args.output_size).cuda(),
            ])

        # Create FFN model
        return nn.Sequential(*ffn)

    def pointwise_feed_forward_to_atom_embedding(self, emb_output, atom_fea, index, ffn_layer):
        aggr_output = select_neighbor_and_aggregate(emb_output, index)
        aggr_outputx = torch.cat([atom_fea.cuda(), aggr_output.cuda()], dim=1)
        return ffn_layer(aggr_outputx), aggr_output

    def forward(self, atom_output, bond_output, original_f_atoms, a2b, a_scope, features_batch):
        rank = dist.get_rank()
        local_rank = int(os.environ['LOCAL_RANK'])
        tag_id = local_rank*100#int((rank//2)*100 + (rank-int(dist.get_world_size()/2))*1000)

        atom_embeddings = self.atom_bond_transform(
                                        atomwise_input=atom_output.cuda(),
                                        bondwise_input=bond_output.cuda(),
                                        original_f_atoms=original_f_atoms,
                                        a2b=a2b,
                                        rank=rank,
                                        tag_id=tag_id)
        atom_in_atom_out, bond_in_atom_out = atom_embeddings

        a_scope = a_scope.data.cpu().numpy().tolist()
        mol_atom_from_bond_output = self.readout(bond_in_atom_out, a_scope)
        
        if features_batch[0] is not None:
            features_batch = torch.from_numpy(np.stack(features_batch)).float()
            if True: # if self.iscuda:
                features_batch = features_batch.cuda()
            features_batch = features_batch.to(atom_in_atom_out)
            if len(features_batch.shape) == 1:
                features_batch = features_batch.view([1, features_batch.shape[0]])
        else:
            features_batch = None

        if features_batch is not None:
            mol_atom_from_bond_output = torch.cat([mol_atom_from_bond_output, features_batch], 1)
        if self.training:
            bond_ffn_output = self.mol_atom_from_bond_ffn(mol_atom_from_bond_output)

            # Send bond_ffn_output
            dist.isend(bond_ffn_output, rank-int(dist.get_world_size()/2), None)

            # Recv atom_ffn_output
            atom_ffn_output = torch.zeros(bond_ffn_output.size(0), self.num_tasks).cuda()
            dist.recv(atom_ffn_output, rank-int(dist.get_world_size()/2), None)
            return atom_ffn_output.cuda(), bond_ffn_output
        else:
            bond_ffn_output = self.mol_atom_from_bond_ffn(mol_atom_from_bond_output)
            if self.classification:
                bond_ffn_output = self.sigmoid(bond_ffn_output)

            # Send bond_ffn_output
            dist.isend(bond_ffn_output, rank-int(dist.get_world_size()/2), None)

            # Recv atom_ffn_output
            atom_ffn_output = torch.zeros(bond_ffn_output.size(0), self.num_tasks).cuda()
            dist.recv(atom_ffn_output, rank-int(dist.get_world_size()/2), None)
            
            output = (atom_ffn_output.cuda() + bond_ffn_output) / 2

        return output



##############################################################################################################
##############################################################################################################
##############################################################################################################
##############################################################################################################


class Encoder_for_PP(nn.Module):
    def __init__(self, model, node_rank, num_layer, is_pipeline_last_stage=False):
        super(Encoder_for_PP, self).__init__()

        self.node_rank = node_rank
        # print(f"len of node_blocks: {len(model.grover.encoders.node_blocks)}")
        count = 0
        for nb in model.grover.encoders.node_blocks:
            if node_rank==count:
                self.node_block = copy.deepcopy(nb.cuda())
            count += 1
        count = 0
        for eb in model.grover.encoders.edge_blocks:
            if node_rank==count:
                self.edge_block = copy.deepcopy(eb.cuda())
            count += 1
        # self.node_block = copy.deepcopy(f"model.grover.encoders.node_blocks.{node_rank}".cuda())

        # if is_pipeline_last_stage:
        #     self.NodeViewReadoutFFN = NodeViewReadoutFFN(model, node_rank)
        #     self.EdgeViewReadoutFFN = EdgeViewReadoutFFN(model, node_rank)
        # else:
        #     self.NodeViewReadoutFFN, self.EdgeViewReadoutFFN = None, None

    def forward(self, f_atoms, f_bonds, a2b, b2a, b2revb, a_scope, b_scope, a2a, features_batch):

        batch = f_atoms, f_bonds, a2b, b2a, b2revb, a_scope, b_scope, a2a

        count = 0
        # for head, head in node_blocks, edge_blocks:
        batch, features_batch = self.node_block(batch, features_batch)
        batch, features_batch = self.edge_block(batch, features_batch)

        node_output, bond_output, _, _, _, _, _, _  = batch

        return node_output, bond_output

    
class AtomInAtomOut(nn.Module):
    def __init__(self, model, rank, args):
        super(AtomInAtomOut, self).__init__()
        self.hidden_size = args.hidden_size
        self.num_tasks = args.num_tasks
        self.embedding_output_type = args.embedding_output_type
        self.ffn_atom_from_atom = copy.deepcopy(model.grover.encoders.ffn_atom_from_atom.cuda())
        self.atom_from_atom_sublayer = copy.deepcopy(model.grover.encoders.atom_from_atom_sublayer.cuda())
        self.ffn_atom_from_bond = copy.deepcopy(model.grover.encoders.ffn_atom_from_bond.cuda())

        self.atom_from_bond_sublayer = copy.deepcopy(model.grover.encoders.atom_from_bond_sublayer.cuda())

        self.act_func_node = copy.deepcopy(model.grover.encoders.act_func_node.cuda())

        self.readout = copy.deepcopy(model.readout.cuda())

        self.mol_atom_from_atom_ffn = copy.deepcopy(model.mol_atom_from_atom_ffn.cuda())

        self.classification = args.dataset_type == 'classification'
        if self.classification:
            self.sigmoid = nn.Sigmoid()

    def atom_bond_transform(self,
                            atomwise_input=None,
                            original_f_atoms=None,
                            a2a=None):
        # atom input to atom output
        atomwise_input, _ = self.pointwise_feed_forward_to_atom_embedding(atomwise_input, original_f_atoms, a2a,
                                                                        self.ffn_atom_from_atom)
        atom_in_atom_out = self.atom_from_atom_sublayer(None, atomwise_input)

        return atom_in_atom_out

    def create_ffn(self, args):
        # Note: args.features_dim is set according the real loaded features data
        if args.features_only:
            first_linear_dim = args.features_size + args.features_dim
        else:
            if args.self_attention:
                first_linear_dim = args.hidden_size * args.attn_out
                # TODO: Ad-hoc!
                # if args.use_input_features:
                first_linear_dim += args.features_dim
            else:
                first_linear_dim = args.hidden_size + args.features_dim

        dropout = nn.Dropout(args.dropout)
        activation = get_activation_function(args.activation)
        # TODO: ffn_hidden_size
        # Create FFN layers
        if args.ffn_num_layers == 1:
            ffn = [
                dropout.cuda(),
                nn.Linear(first_linear_dim, args.output_size).cuda()
            ]
        else:
            ffn = [
                dropout.cuda(),
                nn.Linear(first_linear_dim, args.ffn_hidden_size).cuda()
            ]
            for _ in range(args.ffn_num_layers - 2):
                ffn.extend([
                    activation.cuda(),
                    dropout.cuda(),
                    nn.Linear(args.ffn_hidden_size, args.ffn_hidden_size).cuda(),
                ])
            ffn.extend([
                activation.cuda(),
                dropout.cuda(),
                nn.Linear(args.ffn_hidden_size, args.output_size).cuda(),
            ])

        # Create FFN model
        return nn.Sequential(*ffn)

    def pointwise_feed_forward_to_atom_embedding(self, emb_output, atom_fea, index, ffn_layer):
        aggr_output = select_neighbor_and_aggregate(emb_output, index)
        aggr_outputx = torch.cat([atom_fea.cuda(), aggr_output.cuda()], dim=1)
        return ffn_layer(aggr_outputx), aggr_output

    def forward(self, atom_output, original_f_atoms, a2a, a_scope, features_batch):
        rank = dist.get_rank()
        local_rank = int(os.environ['LOCAL_RANK'])
        atom_embeddings = self.atom_bond_transform(
                                        atomwise_input=atom_output.cuda(),
                                        original_f_atoms=original_f_atoms,
                                        a2a=a2a)
        atom_in_atom_out = atom_embeddings

        a_scope = a_scope.data.cpu().numpy().tolist()
        mol_atom_from_atom_output = self.readout(atom_in_atom_out, a_scope)
        if features_batch[0] is not None:
            features_batch = torch.from_numpy(np.stack(features_batch)).float()
            if 1: # if cuda
                features_batch = features_batch.cuda()
            features_batch = features_batch.to(atom_in_atom_out)
            if len(features_batch.shape) == 1:
                features_batch = features_batch.view([1, features_batch.shape[0]])
        else:
            features_batch = None

        if features_batch is not None:
            mol_atom_from_atom_output = torch.cat([mol_atom_from_atom_output, features_batch], 1)
        if self.training:
            atom_ffn_output = self.mol_atom_from_atom_ffn(mol_atom_from_atom_output)

            return atom_ffn_output
        else:
            atom_ffn_output = self.mol_atom_from_atom_ffn(mol_atom_from_atom_output)
            if self.classification:
                atom_ffn_output = self.sigmoid(atom_ffn_output)

        return atom_ffn_output



class BondInAtomOut(nn.Module):
    def __init__(self, model, rank, args):
        super(BondInAtomOut, self).__init__()
        self.hidden_size = args.hidden_size
        self.num_tasks = args.num_tasks
        self.rank = rank
        self.embedding_output_type = args.embedding_output_type

        self.ffn_atom_from_atom = copy.deepcopy(model.grover.encoders.ffn_atom_from_atom.cuda())
        self.ffn_atom_from_bond = copy.deepcopy(model.grover.encoders.ffn_atom_from_bond.cuda())
        self.atom_from_atom_sublayer = copy.deepcopy(model.grover.encoders.atom_from_atom_sublayer.cuda())
        self.atom_from_bond_sublayer = copy.deepcopy(model.grover.encoders.atom_from_bond_sublayer.cuda())

        self.act_func_edge = copy.deepcopy(model.grover.encoders.act_func_edge.cuda())

        self.readout = copy.deepcopy(model.readout.cuda())

        self.mol_atom_from_bond_ffn = copy.deepcopy(model.mol_atom_from_bond_ffn.cuda())
        
        self.classification = args.dataset_type == 'classification'
        if self.classification:
            self.sigmoid = nn.Sigmoid()

    def atom_bond_transform(self,
                            atomwise_input=None,
                            bondwise_input=None,
                            original_f_atoms=None,
                            a2b=None
                            ):
        # bond to atom
        bondwise_input, _ = self.pointwise_feed_forward_to_atom_embedding(bondwise_input, original_f_atoms, a2b,
                                                                        self.ffn_atom_from_bond)
        bond_in_atom_out = self.atom_from_bond_sublayer(None, bondwise_input)

        return atomwise_input, bond_in_atom_out

    def create_ffn(self, args):
        # Note: args.features_dim is set according the real loaded features data
        if args.features_only:
            first_linear_dim = args.features_size + args.features_dim
        else:
            if args.self_attention:
                first_linear_dim = args.hidden_size * args.attn_out
                # TODO: Ad-hoc!
                # if args.use_input_features:
                first_linear_dim += args.features_dim
            else:
                first_linear_dim = args.hidden_size + args.features_dim

        dropout = nn.Dropout(args.dropout)
        activation = get_activation_function(args.activation)
        # TODO: ffn_hidden_size
        # Create FFN layers
        if args.ffn_num_layers == 1:
            ffn = [
                dropout.cuda(),
                nn.Linear(first_linear_dim, args.output_size).cuda()
            ]
        else:
            ffn = [
                dropout.cuda(),
                nn.Linear(first_linear_dim, args.ffn_hidden_size).cuda()
            ]
            for _ in range(args.ffn_num_layers - 2):
                ffn.extend([
                    activation.cuda(),
                    dropout.cuda(),
                    nn.Linear(args.ffn_hidden_size, args.ffn_hidden_size).cuda(),
                ])
            ffn.extend([
                activation.cuda(),
                dropout.cuda(),
                nn.Linear(args.ffn_hidden_size, args.output_size).cuda(),
            ])

        # Create FFN model
        return nn.Sequential(*ffn)

    def pointwise_feed_forward_to_atom_embedding(self, emb_output, atom_fea, index, ffn_layer):
        aggr_output = select_neighbor_and_aggregate(emb_output, index)
        aggr_outputx = torch.cat([atom_fea.cuda(), aggr_output.cuda()], dim=1)
        return ffn_layer(aggr_outputx), aggr_output

    def forward(self, atom_output, bond_output, original_f_atoms, a2b, a_scope, features_batch):

        atom_embeddings = self.atom_bond_transform(
                                        atomwise_input=atom_output.cuda(),
                                        bondwise_input=bond_output.cuda(),
                                        original_f_atoms=original_f_atoms,
                                        a2b=a2b)
        atom_in_atom_out, bond_in_atom_out = atom_embeddings

        a_scope = a_scope.data.cpu().numpy().tolist()
        mol_atom_from_bond_output = self.readout(bond_in_atom_out, a_scope)
        
        if features_batch[0] is not None:
            features_batch = torch.from_numpy(np.stack(features_batch)).float()
            if True: # if self.iscuda:
                features_batch = features_batch.cuda()
            features_batch = features_batch.to(atom_in_atom_out)
            if len(features_batch.shape) == 1:
                features_batch = features_batch.view([1, features_batch.shape[0]])
        else:
            features_batch = None

        if features_batch is not None:
            mol_atom_from_bond_output = torch.cat([mol_atom_from_bond_output, features_batch], 1)
        if self.training:
            bond_ffn_output = self.mol_atom_from_bond_ffn(mol_atom_from_bond_output)

            return bond_ffn_output
        else:
            bond_ffn_output = self.mol_atom_from_bond_ffn(mol_atom_from_bond_output)
            if self.classification:
                bond_ffn_output = self.sigmoid(bond_ffn_output)

        return bond_ffn_output


class FFN(nn.Module):
    def __init__(self, model, rank, args):
        super(FFN, self).__init__()
        self.hidden_size = args.hidden_size
        self.num_tasks = args.num_tasks
        self.rank = rank
        self.embedding_output_type = args.embedding_output_type

        self.ffn_atom_from_atom = copy.deepcopy(model.grover.encoders.ffn_atom_from_atom.cuda())
        self.ffn_atom_from_bond = copy.deepcopy(model.grover.encoders.ffn_atom_from_bond.cuda())
        self.atom_from_atom_sublayer = copy.deepcopy(model.grover.encoders.atom_from_atom_sublayer.cuda())
        self.atom_from_bond_sublayer = copy.deepcopy(model.grover.encoders.atom_from_bond_sublayer.cuda())

        self.act_func_edge = copy.deepcopy(model.grover.encoders.act_func_edge.cuda())

        self.readout = copy.deepcopy(model.readout.cuda())

        self.mol_atom_from_bond_ffn = copy.deepcopy(model.mol_atom_from_bond_ffn.cuda())
        
        self.classification = args.dataset_type == 'classification'
        if self.classification:
            self.sigmoid = nn.Sigmoid()
    #     self.AtomInAtomOut = AtomInAtomOut(model, rank, args)
    #     self.BondInAtomOut = BondInAtomOut(model, rank, args)

    # def forward(self, atom_output, bond_output, original_f_atoms, a2a, a_scope, features_batch):
    #     print(f"atom_output: {atom_output.shape}")
    #     print(f"original_f_atoms: {original_f_atoms.shape}")
    #     atom_ffn_output = self.AtomInAtomOut(atom_output, original_f_atoms, a2a, a_scope, features_batch)
    #     bond_ffn_output = self.BondInAtomOut(atom_output, bond_output, original_f_atoms, a2b, a_scope, features_batch)

    #     return (atom_ffn_output, bond_ffn_output)/2
    def pointwise_feed_forward_to_atom_embedding(self, emb_output, atom_fea, index, ffn_layer):
        """
        The point-wise feed forward and long-range residual connection for atom view.
        aggregate to atom.
        :param emb_output: the output embedding from the previous multi-head attentions.
        :param atom_fea: the atom/node feature embedding.
        :param index: the index of neighborhood relations.
        :param ffn_layer: the feed forward layer
        :return:
        """
        aggr_output = select_neighbor_and_aggregate(emb_output.cuda(), index)
        aggr_outputx = torch.cat([atom_fea.cuda(), aggr_output], dim=1)
        return ffn_layer(aggr_outputx), aggr_output

    def pointwise_feed_forward_to_bond_embedding(self, emb_output, bond_fea, a2nei, b2revb, ffn_layer):
        """
        The point-wise feed forward and long-range residual connection for bond view.
        aggregate to bond.
        :param emb_output: the output embedding from the previous multi-head attentions.
        :param bond_fea: the bond/edge feature embedding.
        :param index: the index of neighborhood relations.
        :param ffn_layer: the feed forward layer
        :return:
        """
        aggr_output = select_neighbor_and_aggregate(emb_output, a2nei)
        # remove rev bond / atom --- need for bond view
        aggr_output = self.remove_rev_bond_message(emb_output, aggr_output, b2revb)
        aggr_outputx = torch.cat([bond_fea, aggr_output], dim=1)
        return ffn_layer(aggr_outputx), aggr_output

    @staticmethod
    def remove_rev_bond_message(orginal_message, aggr_message, b2revb):
        """

        :param orginal_message:
        :param aggr_message:
        :param b2revb:
        :return:
        """
        rev_message = orginal_message[b2revb]
        return aggr_message - rev_message

    def atom_bond_transform(self,
                            to_atom=True,  # False: to bond
                            atomwise_input=None,
                            bondwise_input=None,
                            original_f_atoms=None,
                            a2a=None,
                            a2b=None
                            ):
        """
        Transfer the output of atom/bond multi-head attention to the final atom/bond output.
        :param to_atom: if true, the output is atom emebedding, otherwise, the output is bond embedding.
        :param atomwise_input: the input embedding of atom/node.
        :param bondwise_input: the input embedding of bond/edge.
        :param original_f_atoms: the initial atom features.
        :param original_f_bonds: the initial bond features.
        :param a2a: mapping from atom index to its neighbors. num_atoms * max_num_bonds
        :param a2b: mapping from atom index to incoming bond indices.
        :param b2a: mapping from bond index to the index of the atom the bond is coming from.
        :param b2revb: mapping from bond index to the index of the reverse bond.
        :return:
        """

        # atom input to atom output
        atomwise_input, _ = self.pointwise_feed_forward_to_atom_embedding(atomwise_input, original_f_atoms, a2a,
                                                                            self.ffn_atom_from_atom)
        atom_in_atom_out = self.atom_from_atom_sublayer(None, atomwise_input)
        # bond to atom
        bondwise_input, _ = self.pointwise_feed_forward_to_atom_embedding(bondwise_input, original_f_atoms, a2b,
                                                                            self.ffn_atom_from_bond)
        bond_in_atom_out = self.atom_from_bond_sublayer(None, bondwise_input)
        return atom_in_atom_out, bond_in_atom_out

    def forward(self, atom_output, bond_output, original_f_atoms, a2a, a2b):

        return self.atom_bond_transform(to_atom=True,  # False: to bond
                                        atomwise_input=atom_output,
                                        bondwise_input=bond_output,
                                        original_f_atoms=original_f_atoms,
                                        a2a=a2a,
                                        a2b=a2b)









################################################################################
# Grover Finetune Task for Pipeline
################################################################################

class GroverFinetuneTask_for_PP(nn.Module):
    """
    The finetune
    """
    def __init__(self, args):
        super(GroverFinetuneTask_for_PP, self).__init__()

        self.hidden_size = args.hidden_size
        self.iscuda = args.cuda

        self.grover = GROVEREmbedding_for_pp(args)

        if args.self_attention:
            self.readout = Readout(rtype="self_attention", hidden_size=self.hidden_size,
                                   attn_hidden=args.attn_hidden,
                                   attn_out=args.attn_out)
        else:
            self.readout = Readout(rtype="mean", hidden_size=self.hidden_size)

        self.mol_atom_from_atom_ffn = self.create_ffn(args)
        self.mol_atom_from_bond_ffn = self.create_ffn(args)

        self.classification = args.dataset_type == 'classification'
        if self.classification:
            self.sigmoid = nn.Sigmoid()
        
        self.node_rank = args.node_rank

    def create_ffn(self, args: Namespace):
        """
        Creates the feed-forward network for the model.

        :param args: Arguments.
        """
        # Note: args.features_dim is set according the real loaded features data
        if args.features_only:
            first_linear_dim = args.features_size + args.features_dim
        else:
            if args.self_attention:
                first_linear_dim = args.hidden_size * args.attn_out
                # TODO: Ad-hoc!
                # if args.use_input_features:
                first_linear_dim += args.features_dim
            else:
                first_linear_dim = args.hidden_size + args.features_dim

        dropout = nn.Dropout(args.dropout)
        activation = get_activation_function(args.activation)
        # TODO: ffn_hidden_size
        # Create FFN layers
        if args.ffn_num_layers == 1:
            ffn = [
                dropout,
                nn.Linear(first_linear_dim, args.output_size)
            ]
        else:
            ffn = [
                dropout,
                nn.Linear(first_linear_dim, args.ffn_hidden_size)
            ]
            for _ in range(args.ffn_num_layers - 2):
                ffn.extend([
                    activation,
                    dropout,
                    nn.Linear(args.ffn_hidden_size, args.ffn_hidden_size),
                ])
            ffn.extend([
                activation,
                dropout,
                nn.Linear(args.ffn_hidden_size, args.output_size),
            ])

        # Create FFN model
        return nn.Sequential(*ffn)

    @staticmethod
    def get_loss_func(args):
        def loss_func(preds, targets,
                      dt=args.dataset_type,
                      dist_coff=args.dist_coff):

            if dt == 'classification':
                pred_loss = nn.BCEWithLogitsLoss(reduction='none')
            elif dt == 'regression':
                pred_loss = nn.MSELoss(reduction='none')
            else:
                raise ValueError(f'Dataset type "{args.dataset_type}" not supported.')

            # print(type(preds))
            # TODO: Here, should we need to involve the model status? Using len(preds) is just a hack.
            if type(preds) is not tuple:
                # in eval mode.
                return pred_loss(preds, targets)

            # in train mode.
            dist_loss = nn.MSELoss(reduction='none')
            # dist_loss = nn.CosineSimilarity(dim=0)
            # print(pred_loss)

            dist = dist_loss(preds[0], preds[1])
            pred_loss1 = pred_loss(preds[0], targets)
            pred_loss2 = pred_loss(preds[1], targets)
            return pred_loss1 + pred_loss2 + dist_coff * dist

        return loss_func

    def forward(self, batch, features_batch):
        _, _, _, _, _, a_scope, _, _, _, _ = batch

        output = self.grover(batch)
        # Share readout
        if self.node_rank == 3:
            mol_atom_from_bond_output = self.readout(output["atom_from_bond"], a_scope)
            mol_atom_from_atom_output = self.readout(output["atom_from_atom"], a_scope)

            if features_batch[0] is not None:
                features_batch = torch.from_numpy(np.stack(features_batch)).float()
                if self.iscuda:
                    features_batch = features_batch.cuda()
                features_batch = features_batch.to(output["atom_from_atom"])
                if len(features_batch.shape) == 1:
                    features_batch = features_batch.view([1, features_batch.shape[0]])
            else:
                features_batch = None


            if features_batch is not None:
                mol_atom_from_atom_output = torch.cat([mol_atom_from_atom_output, features_batch], 1)
                mol_atom_from_bond_output = torch.cat([mol_atom_from_bond_output, features_batch], 1)

            if self.training:
                atom_ffn_output = self.mol_atom_from_atom_ffn(mol_atom_from_atom_output)
                bond_ffn_output = self.mol_atom_from_bond_ffn(mol_atom_from_bond_output)
                return atom_ffn_output, bond_ffn_output
            else:
                atom_ffn_output = self.mol_atom_from_atom_ffn(mol_atom_from_atom_output)
                bond_ffn_output = self.mol_atom_from_bond_ffn(mol_atom_from_bond_output)
                if self.classification:
                    atom_ffn_output = self.sigmoid(atom_ffn_output)
                    bond_ffn_output = self.sigmoid(bond_ffn_output)
                output = (atom_ffn_output + bond_ffn_output) / 2

            return output
        else:
            return output


class GROVEREmbedding_for_pp(nn.Module):
    """
    The GROVER Embedding class. It contains the GTransEncoder.
    This GTransEncoder can be replaced by any validate encoders.
    """

    def __init__(self, args: Namespace):
        """
        Initialize the GROVEREmbedding class.
        :param args:
        """
        super(GROVEREmbedding_for_pp, self).__init__()
        self.embedding_output_type = args.embedding_output_type
        self.node_rank = args.node_rank
        edge_dim = get_bond_fdim() + get_atom_fdim()
        node_dim = get_atom_fdim()
        if not hasattr(args, "backbone"):
            print("No backbone specified in args, use gtrans backbone.")
            args.backbone = "gtrans"
        if args.backbone == "gtrans" or args.backbone == "dualtrans":
            # dualtrans is the old name.
            self.encoders = GTransEncoder_for_pp(args,
                                          hidden_size=args.hidden_size,
                                          edge_fdim=edge_dim,
                                          node_fdim=node_dim,
                                          dropout=args.dropout,
                                          activation=args.activation,
                                          num_mt_block=args.num_mt_block,
                                          num_attn_head=args.num_attn_head,
                                          atom_emb_output=self.embedding_output_type,
                                          bias=args.bias,
                                          cuda=args.cuda)

    def forward(self, graph_batch: List) -> Dict:
        """
        The forward function takes graph_batch as input and output a dict. The content of the dict is decided by
        self.embedding_output_type.

        :param graph_batch: the input graph batch generated by MolCollator.
        :return: a dict containing the embedding results.
        """
        output = self.encoders(graph_batch)
        if self.node_rank == 3:
            return {"atom_from_atom": output[0][0], "bond_from_atom": output[0][1],
                    "atom_from_bond": output[1][0], "bond_from_bond": output[1][1]}
        else:
            return output


class GTransEncoder_for_pp(nn.Module):
    def __init__(self,
                 args,
                 hidden_size,
                 edge_fdim,
                 node_fdim,
                 dropout=0.0,
                 activation="ReLU",
                 num_mt_block=1,
                 num_attn_head=4,
                 atom_emb_output: Union[bool, str] = False,  # options: True, False, None, "atom", "bond", "both"
                 bias=False,
                 cuda=True,
                 res_connection=False):
        """

        :param args: the arguments.
        :param hidden_size: the hidden size of the model.
        :param edge_fdim: the dimension of additional feature for edge/bond.
        :param node_fdim: the dimension of additional feature for node/atom.
        :param dropout: the dropout ratio
        :param activation: the activation function
        :param num_mt_block: the number of mt block.
        :param num_attn_head: the number of attention head.
        :param atom_emb_output:  enable the output aggregation after message passing.
                                              atom_messages:      True                      False
        -False: no aggregating to atom. output size:     (num_atoms, hidden_size)    (num_bonds, hidden_size)
        -True:  aggregating to atom.    output size:     (num_atoms, hidden_size)    (num_atoms, hidden_size)
        -None:                         same as False
        -"atom":                       same as True
        -"bond": aggragating to bond.   output size:     (num_bonds, hidden_size)    (num_bonds, hidden_size)
        -"both": aggregating to atom&bond. output size:  (num_atoms, hidden_size)    (num_bonds, hidden_size)
                                                         (num_bonds, hidden_size)    (num_atoms, hidden_size)
        :param bias: enable bias term in all linear layers.
        :param cuda: run with cuda.
        :param res_connection: enables the skip-connection in MTBlock.
        """
        super(GTransEncoder_for_pp, self).__init__()

        self.node_rank = args.node_rank
        # For the compatibility issue.
        if atom_emb_output is False:
            atom_emb_output = None
        if atom_emb_output is True:
            atom_emb_output = 'atom'

        self.hidden_size = hidden_size
        self.dropout = dropout
        self.activation = activation
        self.cuda = cuda
        self.bias = bias
        self.res_connection = res_connection
        self.edge_blocks = nn.ModuleList()
        self.node_blocks = nn.ModuleList()

        edge_input_dim = edge_fdim
        node_input_dim = node_fdim
        edge_input_dim_i = edge_input_dim
        node_input_dim_i = node_input_dim

        for i in range(num_mt_block):
            if i != 0:
                edge_input_dim_i = self.hidden_size
                node_input_dim_i = self.hidden_size
            self.edge_blocks.append(MTBlock(args=args,
                                            num_attn_head=num_attn_head,
                                            input_dim=edge_input_dim_i,
                                            hidden_size=self.hidden_size,
                                            activation=activation,
                                            dropout=dropout,
                                            bias=self.bias,
                                            atom_messages=False,
                                            cuda=cuda))
            self.node_blocks.append(MTBlock(args=args,
                                            num_attn_head=num_attn_head,
                                            input_dim=node_input_dim_i,
                                            hidden_size=self.hidden_size,
                                            activation=activation,
                                            dropout=dropout,
                                            bias=self.bias,
                                            atom_messages=True,
                                            cuda=cuda))

        self.atom_emb_output = atom_emb_output

        if args.node_rank == 3:
            self.ffn_atom_from_atom = PositionwiseFeedForward(self.hidden_size + node_fdim,
                                                            self.hidden_size * 4,
                                                            activation=self.activation,
                                                            dropout=self.dropout,
                                                            d_out=self.hidden_size)

            self.ffn_atom_from_bond = PositionwiseFeedForward(self.hidden_size + node_fdim,
                                                            self.hidden_size * 4,
                                                            activation=self.activation,
                                                            dropout=self.dropout,
                                                            d_out=self.hidden_size)

            self.ffn_bond_from_atom = PositionwiseFeedForward(self.hidden_size + edge_fdim,
                                                            self.hidden_size * 4,
                                                            activation=self.activation,
                                                            dropout=self.dropout,
                                                            d_out=self.hidden_size)

            self.ffn_bond_from_bond = PositionwiseFeedForward(self.hidden_size + edge_fdim,
                                                            self.hidden_size * 4,
                                                            activation=self.activation,
                                                            dropout=self.dropout,
                                                            d_out=self.hidden_size)

            self.atom_from_atom_sublayer = SublayerConnection(size=self.hidden_size, dropout=self.dropout)
            self.atom_from_bond_sublayer = SublayerConnection(size=self.hidden_size, dropout=self.dropout)
            self.bond_from_atom_sublayer = SublayerConnection(size=self.hidden_size, dropout=self.dropout)
            self.bond_from_bond_sublayer = SublayerConnection(size=self.hidden_size, dropout=self.dropout)

            self.act_func_node = get_activation_function(self.activation)
            self.act_func_edge = get_activation_function(self.activation)

            self.dropout_layer = nn.Dropout(p=args.dropout)

    def pointwise_feed_forward_to_atom_embedding(self, emb_output, atom_fea, index, ffn_layer):
        """
        The point-wise feed forward and long-range residual connection for atom view.
        aggregate to atom.
        :param emb_output: the output embedding from the previous multi-head attentions.
        :param atom_fea: the atom/node feature embedding.
        :param index: the index of neighborhood relations.
        :param ffn_layer: the feed forward layer
        :return:
        """
        aggr_output = select_neighbor_and_aggregate(emb_output, index)
        aggr_outputx = torch.cat([atom_fea, aggr_output], dim=1)
        return ffn_layer(aggr_outputx), aggr_output

    def pointwise_feed_forward_to_bond_embedding(self, emb_output, bond_fea, a2nei, b2revb, ffn_layer):
        """
        The point-wise feed forward and long-range residual connection for bond view.
        aggregate to bond.
        :param emb_output: the output embedding from the previous multi-head attentions.
        :param bond_fea: the bond/edge feature embedding.
        :param index: the index of neighborhood relations.
        :param ffn_layer: the feed forward layer
        :return:
        """
        aggr_output = select_neighbor_and_aggregate(emb_output, a2nei)
        # remove rev bond / atom --- need for bond view
        aggr_output = self.remove_rev_bond_message(emb_output, aggr_output, b2revb)
        aggr_outputx = torch.cat([bond_fea, aggr_output], dim=1)
        return ffn_layer(aggr_outputx), aggr_output

    @staticmethod
    def remove_rev_bond_message(orginal_message, aggr_message, b2revb):
        """

        :param orginal_message:
        :param aggr_message:
        :param b2revb:
        :return:
        """
        rev_message = orginal_message[b2revb]
        return aggr_message - rev_message

    def atom_bond_transform(self,
                            to_atom=True,  # False: to bond
                            atomwise_input=None,
                            bondwise_input=None,
                            original_f_atoms=None,
                            original_f_bonds=None,
                            a2a=None,
                            a2b=None,
                            b2a=None,
                            b2revb=None
                            ):
        """
        Transfer the output of atom/bond multi-head attention to the final atom/bond output.
        :param to_atom: if true, the output is atom emebedding, otherwise, the output is bond embedding.
        :param atomwise_input: the input embedding of atom/node.
        :param bondwise_input: the input embedding of bond/edge.
        :param original_f_atoms: the initial atom features.
        :param original_f_bonds: the initial bond features.
        :param a2a: mapping from atom index to its neighbors. num_atoms * max_num_bonds
        :param a2b: mapping from atom index to incoming bond indices.
        :param b2a: mapping from bond index to the index of the atom the bond is coming from.
        :param b2revb: mapping from bond index to the index of the reverse bond.
        :return:
        """

        if to_atom:
            # atom input to atom output
            atomwise_input, _ = self.pointwise_feed_forward_to_atom_embedding(atomwise_input, original_f_atoms, a2a,
                                                                              self.ffn_atom_from_atom)
            atom_in_atom_out = self.atom_from_atom_sublayer(None, atomwise_input)
            # bond to atom
            bondwise_input, _ = self.pointwise_feed_forward_to_atom_embedding(bondwise_input, original_f_atoms, a2b,
                                                                              self.ffn_atom_from_bond)
            bond_in_atom_out = self.atom_from_bond_sublayer(None, bondwise_input)
            return atom_in_atom_out, bond_in_atom_out
        else:  # to bond embeddings

            # atom input to bond output
            atom_list_for_bond = torch.cat([b2a.unsqueeze(dim=1), a2a[b2a]], dim=1)
            atomwise_input, _ = self.pointwise_feed_forward_to_bond_embedding(atomwise_input, original_f_bonds,
                                                                              atom_list_for_bond,
                                                                              b2a[b2revb], self.ffn_bond_from_atom)
            atom_in_bond_out = self.bond_from_atom_sublayer(None, atomwise_input)
            # bond input to bond output
            bond_list_for_bond = a2b[b2a]
            bondwise_input, _ = self.pointwise_feed_forward_to_bond_embedding(bondwise_input, original_f_bonds,
                                                                              bond_list_for_bond,
                                                                              b2revb, self.ffn_bond_from_bond)
            bond_in_bond_out = self.bond_from_bond_sublayer(None, bondwise_input)
            return atom_in_bond_out, bond_in_bond_out

    def forward(self, batch, features_batch=None):
        f_atoms_original, f_bonds_original, a2b, b2a, b2revb, a_scope, b_scope, a2a, f_atoms_feature, f_bonds_feature = batch
        if self.cuda or next(self.parameters()).is_cuda:
            f_atoms_original, f_bonds_original, a2b, b2a, b2revb = f_atoms_original.cuda(), f_bonds_original.cuda(), a2b.cuda(), b2a.cuda(), b2revb.cuda()
            a2a = a2a.cuda()
        if f_atoms_feature is not None:
            f_atoms_feature, f_bonds_feature = f_atoms_feature.cuda(), f_bonds_feature.cuda()

        
        if f_atoms_feature is not None:
            node_batch = f_atoms_feature, f_bonds_feature, a2b, b2a, b2revb, a_scope, b_scope, a2a
            edge_batch = f_atoms_feature, f_bonds_feature, a2b, b2a, b2revb, a_scope, b_scope, a2a
        else:
            node_batch = f_atoms_original, f_bonds_original, a2b, b2a, b2revb, a_scope, b_scope, a2a
            edge_batch = f_atoms_original, f_bonds_original, a2b, b2a, b2revb, a_scope, b_scope, a2a

        # Note: features_batch is not used here.
        count = 0
        for nb in self.node_blocks:  # atom messages. Multi-headed attention
            if count == self.node_rank:
                node_batch, features_batch = nb(node_batch, features_batch)
            count += 1
        count = 0
        for eb in self.edge_blocks:  # bond messages. Multi-headed attention
            if count == self.node_rank:
                edge_batch, features_batch = eb(edge_batch, features_batch)
            count += 1
        
        atom_output, _, _, _, _, _, _, _ = node_batch  # atom hidden states
        _, bond_output, _, _, _, _, _, _ = edge_batch  # bond hidden states

        if self.node_rank != 3:
            batch = atom_output, bond_output, a2b, b2a, b2revb, a_scope, b_scope, a2a, atom_output, bond_output
            return batch

        atom_embeddings = self.atom_bond_transform(to_atom=True,  # False: to bond
                                                    atomwise_input=atom_output,
                                                    bondwise_input=bond_output,
                                                    original_f_atoms=f_atoms_original,
                                                    original_f_bonds=f_bonds_original,
                                                    a2a=a2a,
                                                    a2b=a2b,
                                                    b2a=b2a,
                                                    b2revb=b2revb)
        bond_embeddings = [None, None]
        # Notice: need to be consistent with output format of DualMPNN encoder
        return ((atom_embeddings[0], bond_embeddings[0]),
                (atom_embeddings[1], bond_embeddings[1]))
