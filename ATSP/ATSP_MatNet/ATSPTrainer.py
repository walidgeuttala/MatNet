
"""
The MIT License

Copyright (c) 2021 MatNet

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.



THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
"""

import torch
from logging import getLogger

from ATSPEnv import ATSPEnv as Env
from ATSPModel import ATSPModel as Model

from torch.optim import Adam as Optimizer
from torch.optim.lr_scheduler import MultiStepLR as Scheduler

from utils.utils import *


class ATSPTrainer:
    def __init__(self,
                 env_params,
                 model_params,
                 optimizer_params,
                 trainer_params):

        # save arguments
        self.env_params = env_params
        self.model_params = model_params
        self.optimizer_params = optimizer_params
        self.trainer_params = trainer_params
        
        # saved data
        self.scaled_problems = None
        self.tour_costs = None

        # result folder, logger
        self.logger = getLogger(name='trainer')
        self.result_folder = get_result_folder()
        self.result_log = LogData()

        # cuda
        USE_CUDA = self.trainer_params['use_cuda']
        if USE_CUDA:
            cuda_device_num = self.trainer_params['cuda_device_num']
            torch.cuda.set_device(cuda_device_num)
            device = torch.device('cuda', cuda_device_num)
            torch.set_default_tensor_type('torch.cuda.FloatTensor')
        else:
            device = torch.device('cpu')
            torch.set_default_tensor_type('torch.FloatTensor')

        # Main Components
        self.model = Model(**self.model_params)
        self.env = Env(**self.env_params)
        self.optimizer = Optimizer(self.model.parameters(), **self.optimizer_params['optimizer'])
        self.scheduler = Scheduler(self.optimizer, **self.optimizer_params['scheduler'])

        # Restore
        self.start_epoch = 1
        model_load = trainer_params['model_load']
        if model_load['enable']:
            checkpoint_fullname = '{path}/checkpoint-{epoch}.pt'.format(**model_load)
            checkpoint = torch.load(checkpoint_fullname, map_location=device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.start_epoch = 1 + model_load['epoch']
            self.result_log.set_raw_data(checkpoint['result_log'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.scheduler.last_epoch = model_load['epoch']-1
            self.logger.info('Saved Model Loaded !!')

        # utility
        self.time_estimator = TimeEstimator()

        # load the data
        self.load_my_problems()

    def run(self):
        self.time_estimator.reset(self.start_epoch)
        for epoch in range(self.start_epoch, self.trainer_params['epochs']+1):
            self.logger.info('=================================================================')

            # LR Decay
            self.scheduler.step()

            # Train
            train_score, train_loss = self._train_one_epoch(epoch)
            self.result_log.append('train_score', epoch, train_score)
            self.result_log.append('train_loss', epoch, train_loss)

            ############################
            # Logs & Checkpoint
            ############################
            elapsed_time_str, remain_time_str = self.time_estimator.get_est_string(epoch, self.trainer_params['epochs'])
            self.logger.info("Epoch {:3d}/{:3d}: Time Est.: Elapsed[{}], Remain[{}]".format(
                epoch, self.trainer_params['epochs'], elapsed_time_str, remain_time_str))

            all_done = (epoch == self.trainer_params['epochs'])
            model_save_interval = self.trainer_params['logging']['model_save_interval']
            img_save_interval = self.trainer_params['logging']['img_save_interval']

            if epoch > 1:  # save latest images, every epoch
                self.logger.info("Saving log_image")
                image_prefix = '{}/latest'.format(self.result_folder)
                util_save_log_image_with_label(image_prefix, self.trainer_params['logging']['log_image_params_1'],
                                    self.result_log, labels=['train_score'])
                util_save_log_image_with_label(image_prefix, self.trainer_params['logging']['log_image_params_2'],
                                    self.result_log, labels=['train_loss'])

            if all_done or (epoch % model_save_interval) == 0:
                self.logger.info("Saving trained_model")
                checkpoint_dict = {
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'scheduler_state_dict': self.scheduler.state_dict(),
                    'result_log': self.result_log.get_raw_data()
                }
                torch.save(checkpoint_dict, '{}/checkpoint-{}.pt'.format(self.result_folder, epoch))

            if all_done or (epoch % img_save_interval) == 0:
                image_prefix = '{}/img/checkpoint-{}'.format(self.result_folder, epoch)
                util_save_log_image_with_label(image_prefix, self.trainer_params['logging']['log_image_params_1'],
                                    self.result_log, labels=['train_score'])
                util_save_log_image_with_label(image_prefix, self.trainer_params['logging']['log_image_params_2'],
                                    self.result_log, labels=['train_loss'])

            if all_done:
                self.logger.info(" *** Training Done *** ")
                self.logger.info("Now, printing log array...")
                util_print_log_array(self.logger, self.result_log)

    def _train_one_epoch(self, epoch):

        score_AM = AverageMeter()
        loss_AM = AverageMeter()
        gap_AM = AverageMeter()

        self.trainer_params['train_episodes'] = len(self.tour_costs)
        train_num_episode = self.trainer_params['train_episodes']
        episode = 0
        loop_cnt = 0
        idx = 0
        while idx < train_num_episode:

            remaining = train_num_episode - episode
            batch_size = min(self.trainer_params['train_batch_size'], remaining)
            idxj = min(idx+batch_size, train_num_episode)
            avg_score, avg_loss = self._train_one_batch(idxj-idx, self.scaled_problems[idx:idxj])
            score_AM.update(avg_score, idxj-idx)
            loss_AM.update(avg_loss, idxj-idx)
            gap_AM.update(self.tour_costs[idx:idxj].float().mean().item(), idxj-idx)
            episode += batch_size

            # Log First 10 Batch, only at the first epoch
            if epoch == self.start_epoch:   
                loop_cnt += 1
                if loop_cnt <= 10:
                    self.logger.info('Epoch {:3d}: Train {:3d}/{:3d}({:1.1f}%)  Score: {:.4f},  Loss: {:.4f}, Gap {:.4f}'
                                     .format(epoch, episode, train_num_episode, 100. * episode / train_num_episode,
                                             score_AM.avg, loss_AM.avg, gap_AM.avg))
                    
            idx += batch_size

        # Log Once, for each epoch
        self.logger.info('Epoch {:3d}: Train ({:3.0f}%)  Score: {:.4f},  Loss: {:.4f}'
                         .format(epoch, 100. * episode / train_num_episode,
                                 score_AM.avg, loss_AM.avg))

        return score_AM.avg, loss_AM.avg

    def _train_one_batch(self, batch_size, problems_batch):

        # Prep
        ###############################################
        self.model.train()
        
        # self.env.load_problems(batch_size)
        self.env.load_problems_manual(problems_batch)
        reset_state, _, _ = self.env.reset()
        self.model.pre_forward(reset_state)

        prob_list = torch.zeros(size=(batch_size, self.env.pomo_size, 0))
        # shape: (batch, pomo, 0~)

        # POMO Rollout
        ###############################################
        state, reward, done = self.env.pre_step()
        while not done:
            selected, prob = self.model(state)
            # shape: (batch, pomo)
            state, reward, done = self.env.step(selected)

            prob_list = torch.cat((prob_list, prob[:, :, None]), dim=2)

        # Loss
        ###############################################
        advantage = reward - reward.float().mean(dim=1, keepdims=True)
        # shape: (batch, pomo)
        log_prob = prob_list.log().sum(dim=2)
        # size = (batch, pomo)
        loss = -advantage * log_prob  # Minus Sign: To Increase REWARD
        # shape: (batch, pomo)
        loss_mean = loss.mean()

        # Score
        ###############################################
        max_pomo_reward, _ = reward.max(dim=1)  # get best results from pomo
        score_mean = -max_pomo_reward.float().mean()  # negative sign to make positive value

        # Step & Return
        ###############################################
        self.model.zero_grad()
        loss_mean.backward()
        self.optimizer.step()
        return score_mean.item(), loss_mean.item()
    

    def load_my_problems(self):
        import torch
        import networkx as nx
        from pathlib import Path
        node_cnt = 64
        # Assuming you have the paths defined properly
        root_dir = Path('../../../atsp_n5900')  # Root directory for instances
        train_file = root_dir / 'train.txt'  # The file containing the instance filenames

        # Read the instance filenames from train.txt
        with open(train_file, 'r') as f:
            instances = f.readlines()
            instances = [x.strip() for x in instances]  # Remove newlines and spaces

        # Prepare tensors to store the adjacency matrices and tour costs
        # Assuming num_instances is the number of instances in train.txt and num_nodes is from each instance graph
        #instances = instances[:10]
        num_instances = len(instances)
        example_graph = nx.read_gpickle(root_dir / instances[0])  # Reading the first graph for size reference
        num_nodes = len(example_graph.nodes)  # Assuming all graphs have the same number of nodes

        # Initialize tensors
        adjacency_matrices = torch.zeros((num_instances, num_nodes, num_nodes))
        tour_costs = torch.zeros(num_instances)
        

        # Iterate over all instances
        for i, instance_file in enumerate(instances):
            # Construct the file path and load the graph using read_gpickle
            graph = nx.read_gpickle(root_dir / instance_file)

            # Extract adjacency matrix
            adj_matrix = nx.to_numpy_array(graph)
            
            # Store adjacency matrix in the tensor
            adjacency_matrices[i] = torch.tensor(adj_matrix)
            
            # Calculate the tour cost by summing the edges with the 'in_solution' attribute
            tour_cost = 0
            for u, v, data in graph.edges(data=True):
                if data.get('in_solution', False):  # Check if the edge is part of the solution
                    tour_cost += data.get('weight', 0)  # Add the edge weight to the tour cost

            # Store the tour cost in the tensor
            tour_costs[i] = tour_cost
        
        adjacency_matrices[:, torch.arange(node_cnt), torch.arange(node_cnt)] = 0
        # Now `adjacency_matrices` holds the adjacency matrices for all instances
        # and `tour_costs` holds the corresponding tour costs.
        
        #scaled_problems =   adjacency_matrices.float() / scaler
        #tour_costs = tour_costs.float() / scaler

        self.scaled_problems = adjacency_matrices
        self.tour_costs = tour_costs
