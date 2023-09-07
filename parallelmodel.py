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

def load_pretrained_model(checkpoint_paths=None): 

    cur_model = 0


    debug(f'Loading model {cur_model} from {args.checkpoint_paths[cur_model]}')
    model = load_checkpoint(checkpoint_paths[cur_model], current_args=None, logger=None)

    return model


class Node_Block_parallel(nn.Module):
    """
    Node block for model parallelism
    """
    def __init__(self, model, rank):
        super(Node_Block_parallel, self).__init__()

        self.rank = rank
        self.node_blocks = copy.deepcopy(model.grover.encoders.node_blocks.cuda())

    def forward(self, f_atoms, f_bonds, a2b, b2a, b2revb, a_scope, b_scope, a2a, features_batch, rank):
        node_batch = f_atoms, f_bonds, a2b, b2a, b2revb, a_scope, b_scope, a2a
        original_f_atoms, original_f_bonds = f_atoms, f_bonds
        
        for nb in self.node_blocks:
            node_batch, features_batch = nb(node_batch, features_batch)

        atom_output, _, _, _, _, _, _, _ = node_batch

        return atom_output.cuda()


class Edge_Block_parallel(nn.Module):
    """
    Node block for model parallelism
    """
    def __init__(self, model, rank):
        super(Edge_Block_parallel, self).__init__()
        self.rank = rank
        self.edge_blocks = copy.deepcopy(model.grover.encoders.edge_blocks.cuda())

    def forward(self, f_atoms, f_bonds, a2b, b2a, b2revb, a_scope, b_scope, a2a, features_batch, rank):
        edge_batch = f_atoms, f_bonds, a2b, b2a, b2revb, a_scope, b_scope, a2a

        for eb in self.edge_blocks:
            edge_batch, features_batch = eb(edge_batch, features_batch)

        _, bond_output, _, _, _, _, _, _  = edge_batch

        return bond_output.cuda()

class Edge_Block_parallel_plus_more_gpu(nn.Module):
    """
    Node block for model parallelism
    """
    def __init__(self, model, rank):
        super(Edge_Block_parallel_plus_more_gpu, self).__init__()
        self.rank = rank
        self.edge_blocks = copy.deepcopy(model.grover.encoders.edge_blocks.cuda())

    def forward(self, f_atoms, f_bonds, a2b, b2a, b2revb, a_scope, b_scope, a2a, features_batch, rank):
        edge_batch = f_atoms, f_bonds, a2b, b2a, b2revb, a_scope, b_scope, a2a

        for eb in self.edge_blocks:
            edge_batch, features_batch = eb(edge_batch, features_batch)

        _, bond_output, _, _, _, _, _, _  = edge_batch

        return bond_output.cuda()

class ReadoutFFN(nn.Module):
    '''
    atom_output, bond_output -> atom_embedding, bond_embedding
    '''
    def __init__(self, model, rank: int, args):
        '''
        :model: pretrained model
        :rank: rank of the process
        '''
        super(ReadoutFFN, self).__init__()
        self.rank = rank
        self.embedding_output_type = args.embedding_output_type
        #self.atom_bond_transform = model.grover.encoders.atom_bond_transform

        self.ffn_atom_from_atom = copy.deepcopy(model.grover.encoders.ffn_atom_from_atom.cuda())
        self.ffn_atom_from_bond = copy.deepcopy(model.grover.encoders.ffn_atom_from_bond.cuda())
        self.ffn_bond_from_atom = copy.deepcopy(model.grover.encoders.ffn_bond_from_atom.cuda())
        self.ffn_bond_from_bond = copy.deepcopy(model.grover.encoders.ffn_bond_from_bond.cuda())

        self.atom_from_atom_sublayer = copy.deepcopy(model.grover.encoders.atom_from_atom_sublayer.cuda())
        self.atom_from_bond_sublayer = copy.deepcopy(model.grover.encoders.atom_from_bond_sublayer.cuda())
        self.bond_from_atom_sublayer = copy.deepcopy(model.grover.encoders.bond_from_atom_sublayer.cuda())
        self.bond_from_bond_sublayer = copy.deepcopy(model.grover.encoders.bond_from_bond_sublayer.cuda())

        self.act_func_node = copy.deepcopy(model.grover.encoders.act_func_node.cuda())
        self.act_func_edge = copy.deepcopy(model.grover.encoders.act_func_edge.cuda())

        self.readout = copy.deepcopy(model.readout)

        self.mol_atom_from_atom_ffn = copy.deepcopy(model.mol_atom_from_atom_ffn.cuda())
        self.mol_atom_from_bond_ffn = copy.deepcopy(model.mol_atom_from_bond_ffn.cuda())

        self.classification = args.dataset_type == 'classification'
        if self.classification:
            self.sigmoid = nn.Sigmoid()

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

    def forward(self, atom_output, bond_output, original_f_atoms, original_f_bonds, a2a, a2b, b2a, b2revb, a_scope, b_scope, features_batch):
        atom_embeddings = self.atom_bond_transform(to_atom=True, 
                                                   atomwise_input=atom_output.cuda(),
                                                   bondwise_input=bond_output.cuda(),
                                                   original_f_atoms=original_f_atoms.cuda(),
                                                   original_f_bonds=original_f_bonds.cuda(),
                                                   a2a=a2a.cuda(),
                                                   a2b=a2b.cuda(),
                                                   b2a=b2a.cuda(),
                                                   b2revb=b2revb.cuda())
        bond_embeddings = self.atom_bond_transform(to_atom=False,
                                                   atomwise_input=atom_output.cuda(),
                                                   bondwise_input=bond_output.cuda(),
                                                   original_f_atoms=original_f_atoms.cuda(),
                                                   original_f_bonds=original_f_bonds.cuda(),
                                                   a2a=a2a.cuda(),
                                                   a2b=a2b.cuda(),
                                                   b2a=b2a.cuda(),
                                                   b2revb=b2revb.cuda())
        output = ((atom_embeddings[0], bond_embeddings[0]),
                  (atom_embeddings[1], bond_embeddings[1]))
        if self.embedding_output_type == 'atom':
            output = {"atom_from_atom": output[0], "atom_from_bond": output[1],
                    "bond_from_atom": None, "bond_from_bond": None}  # atom_from_atom, atom_from_bond
        elif self.embedding_output_type == 'bond':
            output = {"atom_from_atom": None, "atom_from_bond": None,
                    "bond_from_atom": output[0], "bond_from_bond": output[1]}  # bond_from_atom, bond_from_bond
        elif self.embedding_output_type == "both":
            output = {"atom_from_atom": output[0][0], "bond_from_atom": output[0][1],
                    "atom_from_bond": output[1][0], "bond_from_bond": output[1][1]}
        
        a_scope = a_scope.data.cpu().numpy().tolist()
        mol_atom_from_bond_output = self.readout(output["atom_from_bond"].cuda(), a_scope)
        mol_atom_from_atom_output = self.readout(output["atom_from_atom"].cuda(), a_scope)

        if features_batch[0] is not None:
            features_batch = torch.from_numpy(np.stack(features_batch)).float()
            if 1:
                features_batch = features_batch.cuda()
            features_batch = features_batch.to(output["atom_from_atom"])
            if len(features_batch.shape) == 1:
                features_batch = features_batch.view([1, features_batch.shape[0]])
        else:
            features_batch = None

        if features_batch is not None:
            mol_atom_from_atom_output = torch.cat([mol_atom_from_atom_output, features_batch.cuda()], 1)
            mol_atom_from_bond_output = torch.cat([mol_atom_from_bond_output, features_batch.cuda()], 1)
        if self.training:
            atom_ffn_output = self.mol_atom_from_atom_ffn(mol_atom_from_atom_output.cuda())
            bond_ffn_output = self.mol_atom_from_bond_ffn(mol_atom_from_bond_output.cuda())
            return atom_ffn_output, bond_ffn_output
        else:
            atom_ffn_output = self.mol_atom_from_atom_ffn(mol_atom_from_atom_output.cuda())
            bond_ffn_output = self.mol_atom_from_bond_ffn(mol_atom_from_bond_output.cuda())
            if self.classification:
                atom_ffn_output = self.sigmoid(atom_ffn_output.cuda())
                bond_ffn_output = self.sigmoid(bond_ffn_output.cuda())
            output = (atom_ffn_output + bond_ffn_output) / 2

        return output


class Node_Readout_FFN(nn.Module):
    '''
    atom_output, bond_output -> atom_embedding
    '''
    def __init__(self, model, rank, args):
        '''
        :model: pretrained GROVER model
        :args: args
        '''
        super(Node_Readout_FFN, self).__init__()
        self.num_tasks = args.num_tasks
        #self.atom_bond_transform = model.grover.encoders.atom_bond_transform
        self.embedding_output_type = args.embedding_output_type

        self.act_func_node = copy.deepcopy(model.grover.encoders.act_func_node.cuda())

        self.readout = copy.deepcopy(model.readout.cuda())

        self.mol_atom_from_atom_ffn = copy.deepcopy(model.mol_atom_from_atom_ffn.cuda())
        #self.ffn_bond_from_atom = model.grover.encoders.ffn_bond_from_atom.cuda()
        #self.ffn_bond_from_bond = model.grover.encoders.ffn_bond_from_bond.cuda()
        #self.mol_atom_from_atom_ffn = self.create_ffn(args)

        self.classification = args.dataset_type == 'classification'
        if self.classification:
            self.sigmoid = nn.Sigmoid()

    def create_ffn(self, args):
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

    def forward(self, output, a_scope, features_batch):
        rank = dist.get_rank()
        tag_id = int((rank//2)*100)
        
        a_scope = a_scope.data.cpu().numpy().tolist()
        mol_atom_from_atom_output = self.readout(output["atom_from_atom"], a_scope)

        if features_batch[0] is not None:
            features_batch = torch.from_numpy(np.stack(features_batch)).float()
            if 1: # if cuda
                features_batch = features_batch.cuda()
            features_batch = features_batch.to(output["atom_from_atom"])
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
            dist.recv(bond_ffn_output, rank+1, None, tag_id+10)

            # Send atom_ffn_output
            dist.isend(atom_ffn_output, rank+1, None, tag_id+11)

            return atom_ffn_output, bond_ffn_output.cuda()
        else:
            atom_ffn_output = self.mol_atom_from_atom_ffn(mol_atom_from_atom_output)
            if self.classification:
                atom_ffn_output = self.sigmoid(atom_ffn_output)

            # Recv bond_ffn_output
            bond_ffn_output = torch.zeros(atom_ffn_output.size(0), self.num_tasks).cuda()
            dist.recv(bond_ffn_output, rank+1, None, tag_id+10)

            # Send atom_ffn_output
            dist.isend(atom_ffn_output, rank+1, None, tag_id+11)

            output = (atom_ffn_output + bond_ffn_output.cuda()) / 2

        return output


class Node_Readout_FFN2(nn.Module):
    '''
    atom_output, bond_output -> atom_embedding
    '''
    def __init__(self, model, rank, args):
        '''
        :model: pretrained GROVER model
        :args: args
        '''
        super(Node_Readout_FFN2, self).__init__()
        self.hidden_size = args.hidden_size
        self.num_tasks = args.num_tasks
        #self.atom_bond_transform = model.grover.encoders.atom_bond_transform
        self.embedding_output_type = args.embedding_output_type
        self.ffn_atom_from_atom = copy.deepcopy(model.grover.encoders.ffn_atom_from_atom.cuda())
        self.atom_from_atom_sublayer = copy.deepcopy(model.grover.encoders.atom_from_atom_sublayer.cuda())
        self.ffn_atom_from_bond = copy.deepcopy(model.grover.encoders.ffn_atom_from_bond.cuda())

        self.atom_from_bond_sublayer = copy.deepcopy(model.grover.encoders.atom_from_bond_sublayer.cuda())

        self.act_func_node = copy.deepcopy(model.grover.encoders.act_func_node.cuda())

        self.readout = copy.deepcopy(model.readout.cuda())

        self.mol_atom_from_atom_ffn = copy.deepcopy(model.mol_atom_from_atom_ffn.cuda())
        #self.ffn_bond_from_atom = model.grover.encoders.ffn_bond_from_atom.cuda()
        #self.ffn_bond_from_bond = model.grover.encoders.ffn_bond_from_bond.cuda()
        #self.mol_atom_from_atom_ffn = self.create_ffn(args)

        self.classification = args.dataset_type == 'classification'
        if self.classification:
            self.sigmoid = nn.Sigmoid()

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

    def create_ffn(self, args):
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
        aggr_outputx = torch.cat([atom_fea.cuda(), aggr_output.cuda()], dim=1)
        return ffn_layer(aggr_outputx), aggr_output

    def forward(self, atom_output, bond_output, original_f_atoms, original_f_bonds, a2a, a2b, b2a, b2revb, a_scope, features_batch):
        rank = dist.get_rank()
        tag_id = int((rank//2)*100)
        dist.isend(atom_output, rank+1, None, tag_id+0)
        dist.recv(bond_output, rank+1, None, tag_id+1)

        atom_embeddings = self.atom_bond_transform(to_atom=True,  # False: to bond
                                        atomwise_input=atom_output.cuda(),
                                        bondwise_input=bond_output.cuda(),
                                        original_f_atoms=original_f_atoms,
                                        original_f_bonds=original_f_bonds,
                                        a2a=a2a,
                                        a2b=a2b,
                                        b2a=b2a,
                                        b2revb=b2revb)
        # Recv bond_embeddings
        atom_in_bond_out = torch.zeros(original_f_bonds.size(0), self.hidden_size).cuda()
        bond_in_bond_out = torch.zeros(original_f_bonds.size(0), self.hidden_size).cuda()
        dist.recv(atom_in_bond_out, rank+1, None, tag_id+2)
        dist.recv(bond_in_bond_out, rank+1, None, tag_id+3)
        bond_embeddings = (atom_in_bond_out.cuda(), bond_in_bond_out.cuda())

        # Send atom_embeddings
        atom_in_atom_out, bond_in_atom_out = atom_embeddings
        dist.isend(atom_in_atom_out, rank+1, None, tag_id+4)
        dist.isend(bond_in_atom_out, rank+1, None, tag_id+5)

        output = ((atom_embeddings[0], bond_embeddings[0]),
                  (atom_embeddings[1], bond_embeddings[1]))

        if self.embedding_output_type == 'atom':
            output = {"atom_from_atom": output[0], "atom_from_bond": output[1],
                    "bond_from_atom": None, "bond_from_bond": None}  # atom_from_atom, atom_from_bond
        elif self.embedding_output_type == 'bond':
            output = {"atom_from_atom": None, "atom_from_bond": None,
                    "bond_from_atom": output[0], "bond_from_bond": output[1]}  # bond_from_atom, bond_from_bond
        elif self.embedding_output_type == "both":
            output = {"atom_from_atom": output[0][0], "bond_from_atom": output[0][1],
                    "atom_from_bond": output[1][0], "bond_from_bond": output[1][1]}

        a_scope = a_scope.data.cpu().numpy().tolist()
        mol_atom_from_atom_output = self.readout(output["atom_from_atom"], a_scope)

        if features_batch[0] is not None:
            features_batch = torch.from_numpy(np.stack(features_batch)).float()
            if 1: # if cuda
                features_batch = features_batch.cuda()
            features_batch = features_batch.to(output["atom_from_atom"])
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
            dist.recv(bond_ffn_output, rank+1, None, tag_id+10)

            # Send atom_ffn_output
            dist.isend(atom_ffn_output, rank+1, None, tag_id+11)

            return atom_ffn_output, bond_ffn_output.cuda()
        else:
            atom_ffn_output = self.mol_atom_from_atom_ffn(mol_atom_from_atom_output)
            if self.classification:
                atom_ffn_output = self.sigmoid(atom_ffn_output)

            # Recv bond_ffn_output
            bond_ffn_output = torch.zeros(atom_ffn_output.size(0), self.num_tasks).cuda()
            dist.recv(bond_ffn_output, rank+1, None, tag_id+10)

            # Send atom_ffn_output
            dist.isend(atom_ffn_output, rank+1, None, tag_id+11)

            output = (atom_ffn_output + bond_ffn_output.cuda()) / 2

        return output


class Edge_Readout_FFN(nn.Module):
    '''
    atom_output, bond_output -> bond_embedding
    '''
    def __init__(self, model, rank: int, args):
        '''
        :model: pretrained model
        :rank: rank of the process
        '''
        super(Edge_Readout_FFN, self).__init__()
        self.num_tasks = args.num_tasks
        self.rank = rank
        self.embedding_output_type = args.embedding_output_type

        self.act_func_edge = copy.deepcopy(model.grover.encoders.act_func_edge.cuda())

        self.readout = copy.deepcopy(model.readout.cuda())

        self.mol_atom_from_bond_ffn = copy.deepcopy(model.mol_atom_from_bond_ffn.cuda())
        #self.mol_atom_from_bond_ffn = self.create_ffn(args)
        
        self.classification = args.dataset_type == 'classification'
        if self.classification:
            self.sigmoid = nn.Sigmoid()

    def create_ffn(self, args):
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

    def forward(self, output, a_scope, features_batch):
        rank = dist.get_rank()
        tag_id = int((rank//2)*100)
        a_scope = a_scope.data.cpu().numpy().tolist()
        mol_atom_from_bond_output = self.readout(output["atom_from_bond"], a_scope)
        
        if features_batch[0] is not None:
            features_batch = copy.deepcopy(torch.from_numpy(np.stack(features_batch)).float())
            if True: # if self.iscuda:
                features_batch = features_batch.cuda()
            features_batch = features_batch.to(output["atom_from_atom"])
            if len(features_batch.shape) == 1:
                features_batch = features_batch.view([1, features_batch.shape[0]])
        else:
            features_batch = None

        if features_batch is not None:
            mol_atom_from_bond_output = torch.cat([mol_atom_from_bond_output, features_batch], 1)
        if self.training:
            bond_ffn_output = self.mol_atom_from_bond_ffn(mol_atom_from_bond_output)

            # Send bond_ffn_output
            dist.isend(bond_ffn_output, rank-1, None, tag_id+10)

            # Recv atom_ffn_output
            atom_ffn_output = torch.zeros(bond_ffn_output.size(0), self.num_tasks).cuda()
            dist.recv(atom_ffn_output, rank-1, None, tag_id+11)

            return atom_ffn_output.cuda(), bond_ffn_output
        else:
            bond_ffn_output = self.mol_atom_from_bond_ffn(mol_atom_from_bond_output)
            if self.classification:
                bond_ffn_output = self.sigmoid(bond_ffn_output)

            # Send bond_ffn_output
            dist.isend(bond_ffn_output, rank-1, None, tag_id+10)

            # Recv atom_ffn_output
            atom_ffn_output = torch.zeros(bond_ffn_output.size(0), self.num_tasks).cuda()
            dist.recv(atom_ffn_output, rank-1, None, tag_id+11)
            
            output = (atom_ffn_output.cuda() + bond_ffn_output) / 2

        return output


class Edge_Readout_FFN2(nn.Module):
    '''
    atom_output, bond_output -> bond_embedding
    '''
    def __init__(self, model, rank: int, args):
        '''
        :model: pretrained model
        :rank: rank of the process
        '''
        super(Edge_Readout_FFN2, self).__init__()
        self.hidden_size = args.hidden_size
        self.num_tasks = args.num_tasks
        self.rank = rank
        self.embedding_output_type = args.embedding_output_type

        self.bond_from_atom_sublayer = copy.deepcopy(model.grover.encoders.bond_from_atom_sublayer.cuda())
        self.bond_from_bond_sublayer = copy.deepcopy(model.grover.encoders.bond_from_bond_sublayer.cuda())
        
        self.ffn_bond_from_atom = copy.deepcopy(model.grover.encoders.ffn_bond_from_atom.cuda())
        self.ffn_bond_from_bond = copy.deepcopy(model.grover.encoders.ffn_bond_from_bond.cuda())

        self.act_func_edge = copy.deepcopy(model.grover.encoders.act_func_edge.cuda())

        self.readout = copy.deepcopy(model.readout.cuda())

        self.mol_atom_from_bond_ffn = copy.deepcopy(model.mol_atom_from_bond_ffn.cuda())
        #self.mol_atom_from_bond_ffn = self.create_ffn(args)
        
        self.classification = args.dataset_type == 'classification'
        if self.classification:
            self.sigmoid = nn.Sigmoid()

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
        aggr_outputx = torch.cat([bond_fea.cuda(), aggr_output], dim=1)
        return ffn_layer(aggr_outputx), aggr_output

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
        """
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

    def create_ffn(self, args):
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

    def forward(self, atom_output, bond_output, original_f_atoms, original_f_bonds, a2a, a2b, b2a, b2revb, a_scope, features_batch):
        rank = dist.get_rank() # 0, 1
        tag_id = int((rank//2)*100)
        dist.recv(atom_output, rank-1, None, tag_id+0)
        dist.isend(bond_output, rank-1, None, tag_id+1)

        bond_embeddings = self.atom_bond_transform(to_atom=False,  # False: to bond
                                        atomwise_input=atom_output.cuda(),
                                        bondwise_input=bond_output,
                                        original_f_atoms=original_f_atoms,
                                        original_f_bonds=original_f_bonds,
                                        a2a=a2a,
                                        a2b=a2b,
                                        b2a=b2a,
                                        b2revb=b2revb)

        # Send bond embeddings
        atom_in_bond_out, bond_in_bond_out = bond_embeddings[0], bond_embeddings[1]
        dist.isend(atom_in_bond_out, rank-1, None, tag_id+2)
        dist.isend(bond_in_bond_out, rank-1, None, tag_id+3)

        # Recv atom embeddings
        atom_in_atom_out = torch.zeros(original_f_atoms.size(0), self.hidden_size).cuda()
        bond_in_atom_out = torch.zeros(original_f_atoms.size(0), self.hidden_size).cuda()
        dist.recv(atom_in_atom_out, rank-1, None, tag_id+4)
        dist.recv(bond_in_atom_out, rank-1, None, tag_id+5)
        atom_embeddings = (atom_in_atom_out, bond_in_atom_out)

        # Return output
        output = ((atom_embeddings[0], bond_embeddings[0]), # atom in atom out, atom in bond out
                  (atom_embeddings[1], bond_embeddings[1])) # bond in atom out, bond in bond out

        if self.embedding_output_type == 'atom':
            output = {"atom_from_atom": output[0], "atom_from_bond": output[1],
                    "bond_from_atom": None, "bond_from_bond": None}  # atom_from_atom, atom_from_bond
        elif self.embedding_output_type == 'bond':
            output = {"atom_from_atom": None, "atom_from_bond": None,
                    "bond_from_atom": output[0], "bond_from_bond": output[1]}  # bond_from_atom, bond_from_bond
        elif self.embedding_output_type == "both":
            output = {"atom_from_atom": output[0][0], "bond_from_atom": output[0][1],
                    "atom_from_bond": output[1][0], "bond_from_bond": output[1][1]}
        a_scope = a_scope.data.cpu().numpy().tolist()
        mol_atom_from_bond_output = self.readout(output["atom_from_bond"], a_scope)
        
        if features_batch[0] is not None:
            features_batch = torch.from_numpy(np.stack(features_batch)).float()
            if True: # if self.iscuda:
                features_batch = features_batch.cuda()
            features_batch = features_batch.to(output["atom_from_atom"])
            if len(features_batch.shape) == 1:
                features_batch = features_batch.view([1, features_batch.shape[0]])
        else:
            features_batch = None

        if features_batch is not None:
            mol_atom_from_bond_output = torch.cat([mol_atom_from_bond_output, features_batch], 1)
        if self.training:
            bond_ffn_output = self.mol_atom_from_bond_ffn(mol_atom_from_bond_output)

            # Send bond_ffn_output
            dist.isend(bond_ffn_output, rank-1, None, tag_id+10)

            # Recv atom_ffn_output
            atom_ffn_output = torch.zeros(bond_ffn_output.size(0), self.num_tasks).cuda()
            dist.recv(atom_ffn_output, rank-1, None, tag_id+11)
            return atom_ffn_output.cuda(), bond_ffn_output
        else:
            bond_ffn_output = self.mol_atom_from_bond_ffn(mol_atom_from_bond_output)
            if self.classification:
                bond_ffn_output = self.sigmoid(bond_ffn_output)

            # Send bond_ffn_output
            dist.isend(bond_ffn_output, rank-1, None, tag_id+10)

            # Recv atom_ffn_output
            atom_ffn_output = torch.zeros(bond_ffn_output.size(0), self.num_tasks).cuda()
            dist.recv(atom_ffn_output, rank-1, None, tag_id+11)
            
            output = (atom_ffn_output.cuda() + bond_ffn_output) / 2

        return output


class Node_Readout_atom_embedding_only(nn.Module):
    '''
    atom_output, bond_output -> atom_embedding
    '''
    def __init__(self, model, rank, args):
        '''
        :model: pretrained GROVER model
        :args: args
        '''
        super(Node_Readout_atom_embedding_only, self).__init__()
        self.hidden_size = args.hidden_size
        self.num_tasks = args.num_tasks
        #self.atom_bond_transform = model.grover.encoders.atom_bond_transform
        self.embedding_output_type = args.embedding_output_type
        self.ffn_atom_from_atom = copy.deepcopy(model.grover.encoders.ffn_atom_from_atom.cuda())
        self.atom_from_atom_sublayer = copy.deepcopy(model.grover.encoders.atom_from_atom_sublayer.cuda())
        self.ffn_atom_from_bond = copy.deepcopy(model.grover.encoders.ffn_atom_from_bond.cuda())

        self.atom_from_bond_sublayer = copy.deepcopy(model.grover.encoders.atom_from_bond_sublayer.cuda())

        self.act_func_node = copy.deepcopy(model.grover.encoders.act_func_node.cuda())

        self.readout = copy.deepcopy(model.readout.cuda())

        self.mol_atom_from_atom_ffn = copy.deepcopy(model.mol_atom_from_atom_ffn.cuda())
        #self.ffn_bond_from_atom = model.grover.encoders.ffn_bond_from_atom.cuda()
        #self.ffn_bond_from_bond = model.grover.encoders.ffn_bond_from_bond.cuda()
        #self.mol_atom_from_atom_ffn = self.create_ffn(args)

        self.classification = args.dataset_type == 'classification'
        if self.classification:
            self.sigmoid = nn.Sigmoid()

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

    def create_ffn(self, args):
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
        aggr_outputx = torch.cat([atom_fea.cuda(), aggr_output.cuda()], dim=1)
        return ffn_layer(aggr_outputx), aggr_output

    def forward(self, atom_output, bond_output, original_f_atoms, original_f_bonds, a2a, a2b, b2a, b2revb, a_scope, features_batch):
        rank = dist.get_rank()
        tag_id = int((rank//2)*100 + rank*1000)
        dist.isend(atom_output, rank+1, None, tag_id+0)
        dist.recv(bond_output, rank+1, None, tag_id+1)
        atom_embeddings = self.atom_bond_transform(to_atom=True,  # False: to bond
                                        atomwise_input=atom_output.cuda(),
                                        bondwise_input=bond_output.cuda(),
                                        original_f_atoms=original_f_atoms,
                                        original_f_bonds=original_f_bonds,
                                        a2a=a2a,
                                        a2b=a2b,
                                        b2a=b2a,
                                        b2revb=b2revb)
        atom_in_atom_out, bond_in_atom_out = atom_embeddings

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
            dist.recv(bond_ffn_output, rank+1, None, tag_id+10)
            # Send atom_ffn_output
            dist.isend(atom_ffn_output, rank+1, None, tag_id+11)

            return atom_ffn_output, bond_ffn_output.cuda()
        else:
            atom_ffn_output = self.mol_atom_from_atom_ffn(mol_atom_from_atom_output)
            if self.classification:
                atom_ffn_output = self.sigmoid(atom_ffn_output)

            # Recv bond_ffn_output
            bond_ffn_output = torch.zeros(atom_ffn_output.size(0), self.num_tasks).cuda()
            dist.recv(bond_ffn_output, rank+1, None, tag_id+10)
            # Send atom_ffn_output
            dist.isend(atom_ffn_output, rank+1, None, tag_id+11)

            output = (atom_ffn_output + bond_ffn_output.cuda()) / 2

        return output


class Edge_Readout_only_atom_embedding(nn.Module):
    '''
    atom_output, bond_output -> atom_embedding
    '''
    def __init__(self, model, rank: int, args):
        '''
        :model: pretrained model
        :rank: rank of the process
        '''
        super(Edge_Readout_only_atom_embedding, self).__init__()
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
        aggr_outputx = torch.cat([atom_fea.cuda(), aggr_output.cuda()], dim=1)
        return ffn_layer(aggr_outputx), aggr_output

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

    def forward(self, atom_output, bond_output, original_f_atoms, original_f_bonds, a2a, a2b, b2a, b2revb, a_scope, features_batch):
        rank = dist.get_rank()
        tag_id = int((rank//2)*100 + (rank-1)*1000)
        dist.recv(atom_output, rank-1, None, tag_id+0)
        dist.isend(bond_output, rank-1, None, tag_id+1)

        atom_embeddings = self.atom_bond_transform(to_atom=True,  # False: to bond
                                        atomwise_input=atom_output.cuda(),
                                        bondwise_input=bond_output,
                                        original_f_atoms=original_f_atoms,
                                        original_f_bonds=original_f_bonds,
                                        a2a=a2a,
                                        a2b=a2b,
                                        b2a=b2a,
                                        b2revb=b2revb)

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
            dist.isend(bond_ffn_output, rank-1, None, tag_id+10)

            # Recv atom_ffn_output
            atom_ffn_output = torch.zeros(bond_ffn_output.size(0), self.num_tasks).cuda()
            dist.recv(atom_ffn_output, rank-1, None, tag_id+11)
            return atom_ffn_output.cuda(), bond_ffn_output
        else:
            bond_ffn_output = self.mol_atom_from_bond_ffn(mol_atom_from_bond_output)
            if self.classification:
                bond_ffn_output = self.sigmoid(bond_ffn_output)

            # Send bond_ffn_output
            dist.isend(bond_ffn_output, rank-1, None, tag_id+10)

            # Recv atom_ffn_output
            atom_ffn_output = torch.zeros(bond_ffn_output.size(0), self.num_tasks).cuda()
            dist.recv(atom_ffn_output, rank-1, None, tag_id+11)
            
            output = (atom_ffn_output.cuda() + bond_ffn_output) / 2

        return output


class NodeViewReadoutFFN_forEvenRank(nn.Module):
    '''
    atom_output, bond_output -> atom_embedding
    '''
    def __init__(self, model, rank, args):
        '''
        :model: pretrained GROVER model
        :args: args
        '''
        super(NodeViewReadoutFFN_forEvenRank, self).__init__()
        self.hidden_size = args.hidden_size
        self.num_tasks = args.num_tasks
        #self.atom_bond_transform = model.grover.encoders.atom_bond_transform
        self.embedding_output_type = args.embedding_output_type
        self.ffn_atom_from_atom = copy.deepcopy(model.grover.encoders.ffn_atom_from_atom.cuda())
        self.atom_from_atom_sublayer = copy.deepcopy(model.grover.encoders.atom_from_atom_sublayer.cuda())
        self.ffn_atom_from_bond = copy.deepcopy(model.grover.encoders.ffn_atom_from_bond.cuda())

        self.atom_from_bond_sublayer = copy.deepcopy(model.grover.encoders.atom_from_bond_sublayer.cuda())

        self.act_func_node = copy.deepcopy(model.grover.encoders.act_func_node.cuda())

        self.readout = copy.deepcopy(model.readout.cuda())

        self.mol_atom_from_atom_ffn = copy.deepcopy(model.mol_atom_from_atom_ffn.cuda())
        #self.ffn_bond_from_atom = model.grover.encoders.ffn_bond_from_atom.cuda()
        #self.ffn_bond_from_bond = model.grover.encoders.ffn_bond_from_bond.cuda()
        #self.mol_atom_from_atom_ffn = self.create_ffn(args)

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
        """
        """
        # atom input to atom output
        atomwise_input, _ = self.pointwise_feed_forward_to_atom_embedding(atomwise_input, original_f_atoms, a2a,
                                                                        self.ffn_atom_from_atom)
        atom_in_atom_out = self.atom_from_atom_sublayer(None, atomwise_input)

        dist.isend(atom_in_atom_out, rank+1, None, tag_id+0)

        return atom_in_atom_out

    def create_ffn(self, args):
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
        aggr_outputx = torch.cat([atom_fea.cuda(), aggr_output.cuda()], dim=1)
        return ffn_layer(aggr_outputx), aggr_output

    def forward(self, atom_output, original_f_atoms, a2a, a_scope, features_batch):
        rank = dist.get_rank()
        tag_id = int((rank//2)*100 + rank*1000)
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
            dist.recv(bond_ffn_output, rank+1, None, tag_id+10)
            # Send atom_ffn_output
            dist.isend(atom_ffn_output, rank+1, None, tag_id+11)

            return atom_ffn_output, bond_ffn_output.cuda()
        else:
            atom_ffn_output = self.mol_atom_from_atom_ffn(mol_atom_from_atom_output)
            if self.classification:
                atom_ffn_output = self.sigmoid(atom_ffn_output)

            # Recv bond_ffn_output
            bond_ffn_output = torch.zeros(atom_ffn_output.size(0), self.num_tasks).cuda()
            dist.recv(bond_ffn_output, rank+1, None, tag_id+10)
            # Send atom_ffn_output
            dist.isend(atom_ffn_output, rank+1, None, tag_id+11)

            output = (atom_ffn_output + bond_ffn_output.cuda()) / 2

        return output


class NodeViewReadoutFFN_forOddRank(nn.Module):
    '''
    atom_output, bond_output -> atom_embedding
    '''
    def __init__(self, model, rank, args):
        '''
        :model: pretrained GROVER model
        :args: args
        '''
        super(NodeViewReadoutFFN_forOddRank, self).__init__()
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
        """
        """
        # bond to atom
        bondwise_input, _ = self.pointwise_feed_forward_to_atom_embedding(bondwise_input, original_f_atoms, a2b,
                                                                        self.ffn_atom_from_bond)
        bond_in_atom_out = self.atom_from_bond_sublayer(None, bondwise_input)

        dist.recv(atomwise_input, rank-1, None, tag_id+0)
        atomwise_input.detach()

        return atomwise_input, bond_in_atom_out

    def create_ffn(self, args):
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
        aggr_outputx = torch.cat([atom_fea.cuda(), aggr_output.cuda()], dim=1)
        return ffn_layer(aggr_outputx), aggr_output

    def forward(self, atom_output, bond_output, original_f_atoms, a2b, a_scope, features_batch):
        rank = dist.get_rank()
        tag_id = int((rank//2)*100 + (rank-1)*1000)

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
            dist.isend(bond_ffn_output, rank-1, None, tag_id+10)

            # Recv atom_ffn_output
            atom_ffn_output = torch.zeros(bond_ffn_output.size(0), self.num_tasks).cuda()
            dist.recv(atom_ffn_output, rank-1, None, tag_id+11)
            return atom_ffn_output.cuda(), bond_ffn_output
        else:
            bond_ffn_output = self.mol_atom_from_bond_ffn(mol_atom_from_bond_output)
            if self.classification:
                bond_ffn_output = self.sigmoid(bond_ffn_output)

            # Send bond_ffn_output
            dist.isend(bond_ffn_output, rank-1, None, tag_id+10)

            # Recv atom_ffn_output
            atom_ffn_output = torch.zeros(bond_ffn_output.size(0), self.num_tasks).cuda()
            dist.recv(atom_ffn_output, rank-1, None, tag_id+11)
            
            output = (atom_ffn_output.cuda() + bond_ffn_output) / 2

        return output
