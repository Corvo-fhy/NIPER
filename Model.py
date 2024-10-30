import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
import sys
import copy
from utility.helper import *
from utility.batch_test import *
import multiprocessing
import torch.multiprocessing
import random
import pickle


class MyModel(nn.Module):

    def __init__(self, max_item_list, data_config, args):
        super(MyModel, self).__init__()
        # ********************** input data *********************** #
        self.max_item_list = max_item_list
        self.n_users = data_config['n_users']
        self.n_items = data_config['n_items']
        self.num_nodes = self.n_users + self.n_items
        self.pre_adjs = data_config['pre_adjs']
        self.pre_aug = data_config['pre_aug']
        self.pre_adjs_tensor = [self._convert_sp_mat_to_sp_tensor(adj).to(device) for adj in self.pre_adjs]
        self.aug_mat = self._convert_sp_mat_to_sp_tensor(self.pre_aug).to(device)
        self.behs = data_config['behs']
        self.n_relations = len(self.behs)
        # ********************** hyper parameters *********************** #
        self.coefficient = torch.tensor(eval(args.coefficient)).view(1, -1).to(device)
        self.emb_dim = args.embed_size
        self.batch_size = args.batch_size
        self.weight_size = eval(args.layer_size)  # 每一层的输出大小（64）
        self.n_layers = len(self.weight_size)
        self.mess_dropout = eval(args.mess_dropout)  # dropout ratio
        self.aug_type = args.aug_type
        self.nhead = args.nhead
        self.att_dim = args.att_dim
        # ********************** learnable parameters *********************** #
        self.all_weights = {}
        self.all_weights['user_embedding'] = Parameter(torch.FloatTensor(self.n_users, self.emb_dim))
        self.all_weights['item_embedding'] = Parameter(torch.FloatTensor(self.n_items, self.emb_dim))
        self.all_weights['relation_embedding'] = Parameter(torch.FloatTensor(self.n_relations, self.emb_dim))

        self.weight_size_list = [self.emb_dim] + self.weight_size
        # [64, 64, 64, 64, 64]

        for k in range(self.n_layers):
            self.all_weights['W_gc_%d' % k] = Parameter(
                torch.FloatTensor(self.weight_size_list[k], self.weight_size_list[k + 1]))
            self.all_weights['W_rel_%d' % k] = Parameter(
                torch.FloatTensor(self.weight_size_list[k], self.weight_size_list[k + 1]))
        self.reset_parameters()
        self.all_weights = nn.ParameterDict(self.all_weights)
        self.dropout = nn.Dropout(self.mess_dropout[0], inplace=True)
        self.leaky_relu = nn.LeakyReLU(inplace=True)

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.all_weights['user_embedding'])
        nn.init.xavier_uniform_(self.all_weights['item_embedding'])
        nn.init.xavier_uniform_(self.all_weights['relation_embedding'])
        for k in range(self.n_layers):
            nn.init.xavier_uniform_(self.all_weights['W_gc_%d' % k])
            nn.init.xavier_uniform_(self.all_weights['W_rel_%d' % k])

    def _convert_sp_mat_to_sp_tensor(self, X):
        coo = X.tocoo()
        values = coo.data
        indices = np.vstack((coo.row, coo.col))
        shape = coo.shape
        return torch.sparse.FloatTensor(torch.LongTensor(indices), torch.FloatTensor(values), torch.Size(shape))

    def forward(self, device):

        # 表示用户以及物品的Embedding
        ego_embeddings = torch.cat((self.all_weights['user_embedding'], self.all_weights['item_embedding']), dim=0)
        
        ego_aug_embeddings = ego_embeddings
        
        all_rela_embs = {}
        # 取embedding到all_rela_embs中
        for i in range(self.n_relations):
            beh = self.behs[i]
            rela_emb = self.all_weights['relation_embedding'][i]
            rela_emb = torch.reshape(rela_emb, (-1, self.emb_dim))
            all_rela_embs[beh] = [rela_emb]

        total_mm_time = 0.
        all_embeddings = [ego_embeddings]
        all_embeddings_aug = [ego_aug_embeddings]
        for k in range(0, self.n_layers):
            embeddings_list = []
            # 在每个行为中进行
            for i in range(self.n_relations):
                st = time()
                embeddings_ = torch.matmul(self.pre_adjs_tensor[i], ego_embeddings)
                total_mm_time += time() - st
                rela_emb = all_rela_embs[self.behs[i]][k]
                embeddings_ = self.leaky_relu(
                    torch.matmul(torch.mul(embeddings_, rela_emb), self.all_weights['W_gc_%d' % k]))
                embeddings_list.append(embeddings_)
            embeddings_st = torch.stack(embeddings_list, dim=1)

            embeddings = embeddings_st[:, 0, :] * self.coefficient[0][0] + embeddings_st[:, 1, :] * self.coefficient[0][
                1] + embeddings_st[:, 2, :] * self.coefficient[0][2]
            embeddings = self.dropout(embeddings)
            all_embeddings += [embeddings]

            # aug_图
            embeddings_list_aug = []
            # 在每个行为中进行
            for i in range(self.n_relations):
                st = time()
                if i != self.n_relations - 1:
                    embeddings_ = torch.matmul(self.pre_adjs_tensor[i], ego_embeddings)
                else:
                    embeddings_ = torch.matmul(self.aug_mat,
                                               ego_aug_embeddings)
                rela_emb = all_rela_embs[self.behs[i]][k]
                embeddings_ = self.leaky_relu(
                    torch.matmul(torch.mul(embeddings_, rela_emb), self.all_weights['W_gc_%d' % k]))
                embeddings_list_aug.append(embeddings_)
            embeddings_st = torch.stack(embeddings_list_aug, dim=1)

            embeddings = embeddings_st[:, 0, :] * self.coefficient[0][0] + embeddings_st[:, 1, :] * self.coefficient[0][
                1] + embeddings_st[:, 2, :] * self.coefficient[0][2]
            embeddings = self.dropout(embeddings)
            all_embeddings_aug += [embeddings]
            
            for i in range(self.n_relations):
                rela_emb = torch.matmul(all_rela_embs[self.behs[i]][k],
                                        self.all_weights['W_rel_%d' % k])
                all_rela_embs[self.behs[i]].append(rela_emb)

        all_embeddings = torch.stack(all_embeddings, axis=1)
        all_embeddings = torch.mean(all_embeddings, axis=1)
        
        all_embeddings_aug = torch.stack(all_embeddings_aug, axis=1)
        all_embeddings_aug = torch.mean(all_embeddings_aug, axis=1)

        u_g_embeddings, i_g_embeddings = torch.split(all_embeddings, [self.n_users, self.n_items], 0)
        token_embedding = torch.zeros([1, self.emb_dim], device=device)
        i_g_embeddings = torch.cat((i_g_embeddings, token_embedding), dim=0)
        
        u_g_embeddings_aug, i_g_embeddings_aug = torch.split(all_embeddings_aug, [self.n_users, self.n_items], 0)
        i_g_embeddings_aug = torch.cat((i_g_embeddings_aug, token_embedding), dim=0)

        for i in range(self.n_relations):
            all_rela_embs[self.behs[i]] = torch.mean(torch.stack(all_rela_embs[self.behs[i]], 0), 0)

        return u_g_embeddings, i_g_embeddings, all_rela_embs, u_g_embeddings_aug, i_g_embeddings_aug


class RecLoss(nn.Module):
    def __init__(self, data_config, args):
        super(RecLoss, self).__init__()
        self.behs = data_config['behs']
        self.n_relations = len(self.behs)
        self.n_users = data_config['n_users']
        self.n_items = data_config['n_items']
        self.emb_dim = args.embed_size
        self.coefficient = eval(args.coefficient)
        self.wid = eval(args.wid)

    def forward(self, input_u, label_phs, ua_embeddings, ia_embeddings, rela_embeddings):
        uid = ua_embeddings[input_u.long()]
        uid = uid.squeeze()

        pos_r_list = []
        for i in range(self.n_relations):
            beh = self.behs[i]
            pos_beh = ia_embeddings[label_phs[i].long()]  # [B, max_item, dim]
            pos_num_beh = torch.ne(label_phs[i], self.n_items).float()
            pos_beh = torch.einsum('ab,abc->abc', pos_num_beh,
                                   pos_beh)  # [B, max_item] * [B, max_item, dim] -> [B, max_item, dim]
            pos_r = torch.einsum('ac,abc->abc', uid,
                                 pos_beh)  # [B, dim] * [B, max_item, dim] -> [B, max_item, dim]
            pos_r = torch.einsum('ajk,lk->aj', pos_r, rela_embeddings[beh])
            pos_r_list.append(pos_r)

        loss = 0.
        for i in range(self.n_relations):
            beh = self.behs[i]
            # 后半部分
            temp = torch.einsum('ab,ac->bc', ia_embeddings, ia_embeddings) \
                   * torch.einsum('ab,ac->bc', uid, uid)  # [B, dim]' * [B, dim] -> [dim, dim]
            tmp_loss = self.wid[i] * torch.sum(
                temp * torch.matmul(rela_embeddings[beh].T, rela_embeddings[beh]))
            tmp_loss += torch.sum((1.0 - self.wid[i]) * torch.square(pos_r_list[i]) - 2.0 * pos_r_list[i])

            loss += self.coefficient[i] * tmp_loss

        regularizer = torch.sum(torch.square(uid)) * 0.5 + torch.sum(torch.square(ia_embeddings)) * 0.5
        emb_loss = args.decay * regularizer

        return loss, emb_loss


def preprocess_sim(config):
    
    user = sp.load_npz(f"Sim/{config['dataset']}_user_matrix.npz")
    user_indices = torch.tensor(user.toarray()).bool()

    item = sp.load_npz(f"Sim/{config['dataset']}_item_matrix.npz")
    item_indices = torch.tensor(item.toarray()).bool()

    return user_indices, item_indices



def get_lables(temp_set, k=0.9999):
    max_item = 0
    item_lenth = []
    for i in temp_set:
        item_lenth.append(len(temp_set[i]))
        if len(temp_set[i]) > max_item:
            max_item = len(temp_set[i])
    item_lenth.sort()

    max_item = item_lenth[int(len(item_lenth) * k) - 1]

    print(max_item)
    for i in temp_set:
        if len(temp_set[i]) > max_item:
            temp_set[i] = temp_set[i][0:max_item]
        while len(temp_set[i]) < max_item:
            temp_set[i].append(n_items)
    return max_item, temp_set


def get_train_instances1(max_item_list, beh_label_list):
    user_train = []
    beh_item_list = [list() for i in range(n_behs)]  #

    for i in beh_label_list[-1].keys():
        user_train.append(i)
        beh_item_list[-1].append(beh_label_list[-1][i])
        for j in range(n_behs - 1):
            if not i in beh_label_list[j].keys():
                beh_item_list[j].append([n_items] * max_item_list[j])
            else:
                beh_item_list[j].append(beh_label_list[j][i])

    user_train = np.array(user_train)
    beh_item_list = [np.array(beh_item) for beh_item in beh_item_list]
    user_train = user_train[:, np.newaxis]
    return user_train, beh_item_list


def get_train_pairs(user_train_batch, beh_item_tgt_batch):
    input_u_list, input_i_list = [], []
    for i in range(len(user_train_batch)):
        pos_items = beh_item_tgt_batch[i][np.where(beh_item_tgt_batch[i] != n_items)]  # ndarray [x,]
        uid = user_train_batch[i][0]
        input_u_list += [uid] * len(pos_items)
        input_i_list += pos_items.tolist()

    return np.array(input_u_list).reshape([-1]), np.array(input_i_list).reshape([-1])


def test_torch(ua_embeddings, ia_embeddings, rela_embedding, users_to_test, batch_test_flag=False):
    def get_score_np(ua_embeddings, ia_embeddings, rela_embedding, users, items):
        ug_embeddings = ua_embeddings[users]  # []
        pos_ig_embeddings = ia_embeddings[items]
        dot = np.multiply(pos_ig_embeddings, rela_embedding)  # [I, dim] * [1, dim]-> [I, dim]
        batch_ratings = np.matmul(ug_embeddings, dot.T)  # [U, dim] * [dim, I] -> [U, I]
        return batch_ratings

    result = {'precision': np.zeros(len(Ks)), 'recall': np.zeros(len(Ks)), 'ndcg': np.zeros(len(Ks)),
              'hit_ratio': np.zeros(len(Ks)), 'auc': 0.}

    test_users = users_to_test
    n_test_users = len(test_users)

    # pool = torch.multiprocessing.Pool(cores)
    pool = multiprocessing.Pool(cores)

    u_batch_size = BATCH_SIZE
    n_user_batchs = n_test_users // u_batch_size + 1

    count = 0
    for u_batch_id in range(n_user_batchs):
        start = u_batch_id * u_batch_size
        end = (u_batch_id + 1) * u_batch_size

        user_batch = test_users[start: end]

        item_batch = range(ITEM_NUM)
        rate_batch = get_score_np(ua_embeddings, ia_embeddings, rela_embedding, user_batch, item_batch)

        user_batch_rating_uid = zip(rate_batch, user_batch)
        batch_result = pool.map(test_one_user, user_batch_rating_uid)
        count += len(batch_result)

        for re in batch_result:
            result['precision'] += re['precision'] / n_test_users
            result['recall'] += re['recall'] / n_test_users
            result['ndcg'] += re['ndcg'] / n_test_users
            result['hit_ratio'] += re['hit_ratio'] / n_test_users
            result['auc'] += re['auc'] / n_test_users
    assert count == n_test_users

    pool.close()
    return result


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    np.random.seed(seed)  # Numpy module.
    random.seed(seed)  # Python random module.

    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    os.environ['PYTHONHASHSEED'] = str(seed)


if __name__ == '__main__':
    torch.autograd.set_detect_anomaly(True)
    torch.cuda.set_device(0)
    os.environ["GIT_PYTHON_REFRESH"] = "quiet"
    os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_seed(2020)

    config = dict()
    config['device'] = device
    config['n_users'] = data_generator.n_users
    config['n_items'] = data_generator.n_items
    config['behs'] = data_generator.behs
    config['dataset'] = data_generator.dataset_name

    config['trn_mat'] = data_generator.trnMats[-1]  
    
    data_info = [
                "wid : %s" % (args.wid),
                 "decay: %0.2f" % (args.decay),
                 "coefficient : %s" % (args.coefficient)
    ]
    data_info = "\n".join(data_info)
    print(data_info)


    
    pre_adj_list, pre_adj_aug = data_generator.get_adj_mat()
    config['pre_adjs'] = pre_adj_list
    config['pre_aug'] = pre_adj_aug
    print('use the pre adjcency matrix')
    n_users, n_items = data_generator.n_users, data_generator.n_items
    behs = data_generator.behs
    n_behs = data_generator.beh_num
    
    user_indices, item_indices = preprocess_sim(config)

    trnDicts = copy.deepcopy(data_generator.trnDicts)
    max_item_list = []
    beh_label_list = []

    
    for i in range(n_behs):
        max_item, beh_label = get_lables(trnDicts[i])
        max_item_list.append(max_item)
        beh_label_list.append(beh_label)

    t0 = time()

    
    model = MyModel(max_item_list, data_config=config, args=args).to(device)
    recloss = RecLoss(data_config=config, args=args).to(device)

    
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.lr_decay_step, gamma=args.lr_gamma)
    cur_best_pre_0 = 0.
    print('without pretraining.')

    run_time = 1

    loss_loger, pre_loger, rec_loger, ndcg_loger, hit_loger = [], [], [], [], []

    stopping_step = 0
    should_stop = False

    
    user_train1, beh_item_list = get_train_instances1(max_item_list, beh_label_list)

    nonshared_idx = -1

    
    for epoch in range(args.epoch):
        
        model.train()
        shuffle_indices = np.random.permutation(np.arange(len(user_train1)))
        user_train1 = user_train1[shuffle_indices]
        beh_item_list = [beh_item[shuffle_indices] for beh_item in beh_item_list]
        t1 = time()
        loss, rec_loss, emb_loss, ssl_loss, ssl2_loss = 0., 0., 0., 0., 0.
        n_batch = int(len(user_train1) / args.batch_size)

        iter_time = time()

        
        for idx in range(n_batch):
            optimizer.zero_grad()

            start_index = idx * args.batch_size
            end_index = min((idx + 1) * args.batch_size, len(user_train1))

            u_batch = user_train1[start_index:end_index]
            beh_batch = [beh_item[start_index:end_index] for beh_item in
                         beh_item_list]  # [[Batch, max_item1], [Batch, max_item2], [Batch, max_item3]]

            
            
            u_batch_list, i_batch_list = get_train_pairs(user_train_batch=u_batch,
                                                         beh_item_tgt_batch=beh_batch[-1])
            
            # load into cuda
            u_batch = torch.from_numpy(u_batch).to(device)
            beh_batch = [torch.from_numpy(beh_item).to(device) for beh_item in beh_batch]
            u_batch_list = torch.from_numpy(u_batch_list).to(device)
            i_batch_list = torch.from_numpy(i_batch_list).to(device)

            u_batch_indices = user_indices[u_batch_list.type(torch.long)].to(device)
            i_batch_indices = item_indices[i_batch_list.type(torch.long)].to(device)
            
            model_time = time()

            ua_embeddings, ia_embeddings, rela_embeddings, ua_embeddings_aug, ia_embeddings_aug = model(device)
            
            # ------------------------------------------------------------------------------------------------
            
            ssl_temp = 0.1
            # 
            batch_intra_ssl_loss = 0.
            alpha = 0.05
            emb_tgt = torch.multiply(ua_embeddings, rela_embeddings['train'])[u_batch_list.type(torch.long)]
            normalize_emb_tgt = F.normalize(emb_tgt, dim=1)
            normalize_all_emb_aux = F.normalize(torch.multiply(ua_embeddings, rela_embeddings['train']), dim=1)
            ttl_score = torch.matmul(normalize_emb_tgt, normalize_all_emb_aux.T)  # [B, N]
            pos_score = ttl_score.clone()
            pos_score[~u_batch_indices] = 0.
            ttl_score[u_batch_indices] = 0.
            pos_score = torch.sum(torch.exp(pos_score / ssl_temp), dim=1)
            ttl_score = torch.sum(torch.exp(ttl_score / ssl_temp), dim=1)
            batch_intra_ssl_loss += -torch.sum(torch.log(pos_score / ttl_score)) 
            
            batch_intra_ssl_loss = batch_intra_ssl_loss * alpha            
            # ------------------------------------------------------------------------------------------------
            # 
            
            batch_inter_ssl_loss = 0.
            user_emb1 = ua_embeddings[u_batch_list]
            user_emb2 = ua_embeddings_aug[u_batch_list]  # [B, dim]
            normalize_user_emb1 = F.normalize(user_emb1, dim=1)
            normalize_user_emb2 = F.normalize(user_emb2, dim=1)
            normalize_all_user_emb2 = F.normalize(ua_embeddings_aug, dim=1)
            pos_score_user = torch.sum(torch.mul(normalize_user_emb1, normalize_user_emb2),
                                       dim=1)
            pos_score_user = torch.exp(pos_score_user / ssl_temp)

            ttl_score_user = torch.matmul(normalize_user_emb1,
                                          normalize_all_user_emb2.T)
            ttl_score_user = torch.sum(torch.exp(ttl_score_user / ssl_temp), dim=1)  # [B, ]

            ssl_loss_user = -torch.sum(torch.log(pos_score_user / ttl_score_user))
            
            item_emb1 = ia_embeddings[i_batch_list]
            item_emb2 = ia_embeddings_aug[i_batch_list]

            normalize_item_emb1 = F.normalize(item_emb1, dim=1)
            normalize_item_emb2 = F.normalize(item_emb2, dim=1)
            normalize_all_item_emb2 = F.normalize(ia_embeddings_aug, dim=1)
            pos_score_item = torch.sum(torch.mul(normalize_item_emb1, normalize_item_emb2), dim=1)
            ttl_score_item = torch.matmul(normalize_item_emb1, normalize_all_item_emb2.T)

            pos_score_item = torch.exp(pos_score_item / ssl_temp)
            ttl_score_item = torch.sum(torch.exp(ttl_score_item / ssl_temp), dim=1)

            ssl_loss_item = -torch.sum(torch.log(pos_score_item / ttl_score_item))
            
            batch_inter_ssl_loss =  (ssl_loss_user + ssl_loss_item) * 0.05

            #--------------------------------------------------------------------------------------------------
            

            batch_rec_loss, batch_emb_loss = recloss(u_batch, beh_batch, ua_embeddings, ia_embeddings, rela_embeddings)

            batch_loss = batch_rec_loss + batch_emb_loss + batch_inter_ssl_loss + batch_intra_ssl_loss

            batch_loss.backward()
            optimizer.step()

            loss += batch_loss.item() / n_batch
            rec_loss += batch_rec_loss.item() / n_batch
            emb_loss += batch_emb_loss.item() / n_batch
            ssl_loss += batch_intra_ssl_loss.item() / n_batch
            ssl2_loss += batch_inter_ssl_loss.item() / n_batch

        if args.lr_decay: scheduler.step()
        torch.cuda.empty_cache()

        if np.isnan(loss) == True:
            print('ERROR: loss is nan.')
            sys.exit()

        # print the test evaluation metrics each 10 epochs; pos:neg = 1:10.
        if (epoch + 1) % args.test_epoch != 0:
            if args.verbose > 0 and epoch % args.verbose == 0:
                perf_str = 'Epoch %d [%.1fs]: train==[%.5f=%.5f + %.5f + %.5f + %.5f]' % (
                    epoch, time() - t1, loss, rec_loss, emb_loss, ssl_loss, ssl2_loss)
                print(perf_str)
            continue

        t2 = time()
        model.eval()
        # 测试模型
        with torch.no_grad():
            ua_embeddings, ia_embeddings, rela_embeddings, _, _ = model(device)
            users_to_test = list(data_generator.test_set.keys())

            ret = test_torch(ua_embeddings.detach().cpu().numpy(),
                             ia_embeddings.detach().cpu().numpy(),
                             rela_embeddings[behs[-1]].detach().cpu().numpy(), users_to_test)

        t3 = time()

        loss_loger.append(loss)
        rec_loger.append(ret['recall'])
        pre_loger.append(ret['precision'])
        ndcg_loger.append(ret['ndcg'])
        hit_loger.append(ret['hit_ratio'])

        if args.verbose > 0:
            perf_str = 'Epoch %d [%.1fs + %.1fs]:, recall=[%.5f, %.5f], ' \
                       'precision=[%.5f, %.5f], hit=[%.5f, %.5f], ndcg=[%.5f, %.5f]' % \
                       (
                           epoch, t2 - t1, t3 - t2, ret['recall'][0],
                           ret['recall'][1],
                           ret['precision'][0], ret['precision'][1], ret['hit_ratio'][0], ret['hit_ratio'][1],
                           ret['ndcg'][0], ret['ndcg'][1])
            print(perf_str)

        cur_best_pre_0, stopping_step, should_stop, flag = early_stopping_new(ret['recall'][0], cur_best_pre_0,
                                                                              stopping_step, expected_order='acc',
                                                                              flag_step=10)
        # *********************************************************
        # early stopping when cur_best_pre_0 is decreasing for ten successive steps.
        if should_stop == True:
            break

    recs = np.array(rec_loger)
    pres = np.array(pre_loger)
    ndcgs = np.array(ndcg_loger)
    hit = np.array(hit_loger)

    best_rec_0 = max(recs[:, 0])
    idx = list(recs[:, 0]).index(best_rec_0)

    final_perf = "Best Iter=[%d]@[%.1f]\trecall=[%s], ndcg=[%s]" % \
                 (idx, time() - t0, '\t'.join(['%.4f' % r for r in recs[idx]]),
                  '\t'.join(['%.4f' % r for r in ndcgs[idx]]))
    print(final_perf)
    save_dict = f"{data_generator.dataset_name}.pth.tar"
    torch.save(model.state_dict(), save_dict)
    print(f"Model Save To {save_dict}.pth.tar")

