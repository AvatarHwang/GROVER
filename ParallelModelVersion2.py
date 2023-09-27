'''
Models for Model Parallel
'''

import torch
from torch import nn
from grover.util.utils import load_checkpoint
from grover.model.layers import Readout
import numpy as np

from grover.util.nn_utils import get_activation_function, select_neighbor_and_aggregate

import copy
from torch import distributed as dist

import time
import os


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
        if num_layer == 1:
            self.node_blocks = copy.deepcopy(model.grover.encoders.node_blocks.cuda())
            self.edge_blocks = copy.deepcopy(model.grover.encoders.edge_blocks.cuda())
        
        else:
            self.node_blocks = [copy.deepcopy(model.grover.encoders.node_blocks.heads[i].cuda()) for i in range(node_rank*2, node_rank*2+2)]
            self.edge_blocks = [copy.deepcopy(model.grover.encoders.edge_blocks.heads[i].cuda()) for i in range(node_rank*2, node_rank*2+2)]
        # if is_pipeline_last_stage:
        #     self.NodeViewReadoutFFN = NodeViewReadoutFFN(model, node_rank)
        #     self.EdgeViewReadoutFFN = EdgeViewReadoutFFN(model, node_rank)
        # else:
        #     self.NodeViewReadoutFFN, self.EdgeViewReadoutFFN = None, None

    def forward(self, f_atoms, f_bonds, a2b, b2a, b2revb, a_scope, b_scope, a2a, features_batch):

        # f_atoms, f_bonds, a2b, b2a, b2revb, a_scope, b_scope, a2a = batch

        if self.atom_messages:
            # Only add linear transformation in the input feature.
            if f_atoms.shape[1] != self.hidden_size:
                self.W_i = self.W_i.cuda()
                f_atoms = self.W_i(f_atoms.cuda())
                f_atoms = self.dropout_layer(self.layernorm(self.act_func(f_atoms)))

        else:  # bond messages
            if f_bonds.shape[1] != self.hidden_size:
                self.W_i = self.W_i.cuda()
                f_bonds = self.W_i(f_bonds.cuda())
                f_bonds = self.dropout_layer(self.layernorm(self.act_func(f_bonds)))

        queries = []
        keys = []
        values = []
        head = heads[self.node_rank]
        q, k, v = head(f_atoms, f_bonds, a2b, a2a, b2a, b2revb)
        queries.append(q.unsqueeze(1))
        keys.append(k.unsqueeze(1))
        values.append(v.unsqueeze(1))
        queries = torch.cat(queries, dim=1)
        keys = torch.cat(keys, dim=1)
        values = torch.cat(values, dim=1)

        x_out = self.attn(queries, keys, values)  # multi-headed attention
        x_out = x_out.view(x_out.shape[0], -1)
        x_out = self.W_o(x_out)

        x_in = None
        # support no residual connection in MTBlock.
        if self.res_connection:
            if self.atom_messages:
                x_in = f_atoms
            else:
                x_in = f_bonds

        if self.atom_messages:
            f_atoms = self.sublayer(x_in, x_out)
        else:
            f_bonds = self.sublayer(x_in, x_out)

        batch = f_atoms, f_bonds, a2b, b2a, b2revb, a_scope, b_scope, a2a
        features_batch = features_batch
        return batch, features_batch

    
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
                            a2a=None
                            ):
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
                            a2b=None,
                            rank=0,
                            tag_id=0
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

            return bond_ffn_output
        else:
            bond_ffn_output = self.mol_atom_from_bond_ffn(mol_atom_from_bond_output)
            if self.classification:
                bond_ffn_output = self.sigmoid(bond_ffn_output)

        return bond_ffn_output


class FFN(nn.Module):
    def __init__(self, model, rank, args):
        super(FFN, self).__init__()
        self.AtomInAtomOut = AtomInAtomOut(model, rank, args)
        self.BondInAtomOut = BondInAtomOut(model, rank, args)

    def forward(self, atom_output, bond_output, original_f_atoms, a2a, a_scope, features_batch):
        atom_ffn_output = self.AtomInAtomOut(atom_output, original_f_atoms, a2a, a_scope, features_batch)
        bond_ffn_output = self.BondInAtomOut(atom_output, bond_output, original_f_atoms, a2b, a_scope, features_batch)

        return (atom_ffn_output, bond_ffn_output)/2