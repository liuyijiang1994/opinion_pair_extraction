import argparse
from model import NER_NET
from data_loader import create_batch_iter
import torch
import common
import time
from util import time_since
from transformers import AdamW, get_linear_schedule_with_warmup
from Logginger import init_logger
from score import eval_result, eval_rel_by_condition
import sys

logger = init_logger("torch", logging_path=common.log_path)


def train_model(model, optimizer, scheduler, train_iter, test_iter, opt, len_dataset):
    print('======================  Start Training  =========================')
    best_f1 = 0
    global_step = 0
    patience = opt.patience
    for e in range(opt.num_epoch):
        if patience <= 0:
            break
        total_loss = 0
        epoch_start = time.time()
        temp_start = epoch_start
        model.train()
        for step, batch in enumerate(train_iter):
            words, pieces, batch = batch
            batch = tuple(t.to(opt.device) for t in batch)
            word_span, word_mask, poses_ids, labels_id, pieces_ids, pieces_mask, segment_ids, relations, adj = batch
            word_encode, gcn_word_encode, word_logits = model(poses_ids, pieces_ids, pieces_mask, segment_ids,
                                                              word_span, adj)
            ner_loss, rel_loss = model.loss(word_encode, gcn_word_encode, labels_id, word_logits, word_mask, relations)
            rel_loss = opt.rel_loss_weight * rel_loss
            train_loss = ner_loss + rel_loss
            total_loss += train_loss.item()
            train_loss.backward()
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

            global_step += 1
            labels_id = labels_id.cpu()
            predict_labels, rel_output = model.predict(word_encode, gcn_word_encode, word_logits, word_mask)
            train_reporter = eval_result(predict_labels, labels_id)
            triple_p, triple_r, triple_f, ovrlap_recall = eval_rel_by_condition(
                rel_output.detach().cpu().numpy(),
                predict_labels,
                relations.cpu().numpy(),
                labels_id)

            if step % opt.print_every == 0:
                temp_time = time.time()
                temp_cost = temp_time - temp_start
                temp_start = temp_time
                print(
                    "   Instance: %s; Time: %.2fs; loss: %.4f | %.4f ; OT: %.4f %.4f %.4f ; OW: %.4f %.4f %.4f ; R: %.4f %.4f %.4f ; ovrlap : %.4f" % (
                        step * opt.train_batch_size,
                        temp_cost,
                        ner_loss,
                        rel_loss,
                        train_reporter['OT_p'],
                        train_reporter['OT_r'],
                        train_reporter['OT_f'],
                        train_reporter['OW_p'],
                        train_reporter['OW_r'],
                        train_reporter['OW_f'],
                        triple_p, triple_r, triple_f, ovrlap_recall))

                sys.stdout.flush()

        model.eval()
        count = 0
        eval_predict_labels, y_labels = [], []
        all_eval_loss = 0
        eval_rel_tar_list = []
        eval_rel_pre_list = []
        with torch.no_grad():
            test_start = time.time()
            for step, batch in enumerate(test_iter):
                words, pieces, batch = batch
                batch = tuple(t.to(opt.device) for t in batch)
                word_span, word_mask, poses_ids, labels_id, pieces_ids, pieces_mask, segment_ids, relations, adj = batch
                word_encode, gcn_word_encode, word_logits = model(poses_ids, pieces_ids, pieces_mask, segment_ids,
                                                                  word_span, adj)
                eval_ner_loss, eval_rel_loss = model.loss(word_encode, gcn_word_encode, labels_id, word_logits,
                                                          word_mask, relations)
                eval_rel_loss = opt.rel_loss_weight * eval_rel_loss
                eval_loss = eval_ner_loss + eval_rel_loss
                all_eval_loss += eval_loss.item()
                predict_labels, rel_output = model.predict(word_encode, gcn_word_encode, word_logits, word_mask)

                eval_predict_labels.extend(predict_labels)
                eval_rel_tar_list.append(relations.cpu())
                eval_rel_pre_list.append(rel_output.cpu())
                y_labels.append(labels_id.cpu())

                count += 1

            eval_labeled = torch.cat(y_labels, dim=0)
            eval_rel_tar_list = torch.cat(eval_rel_tar_list, dim=0)
            eval_rel_pre_list = torch.cat(eval_rel_pre_list, dim=0)
            eval_reporter = eval_result(eval_predict_labels, eval_labeled, show=True)
            eval_triple_p, eval_triple_r, eval_triple_f, ovrlap_recall = eval_rel_by_condition(
                eval_rel_pre_list.detach().cpu().numpy(),
                eval_predict_labels,
                eval_rel_tar_list.cpu().numpy(),
                eval_labeled)

        if eval_triple_f >= best_f1:
            print('best performance, saving model...')
            best_f1 = eval_triple_f
            torch.save(model.state_dict(), common.output_dir + '/model.pt')

        test_finish = time.time()
        test_cost = test_finish - test_start
        epoch_finish = time.time()
        epoch_cost = epoch_finish - epoch_start
        print()
        print("Epoch: %s training finished. Time: %.2fs, speed: %.2fst/s,  total loss: %s | %s " % (
            e, epoch_cost, len_dataset / epoch_cost, total_loss, all_eval_loss))
        print("test: time: %.2fs, speed: %.2fst/s loss: %.4f | %.4f" % (
            test_cost, 0, eval_ner_loss, eval_rel_loss))
        print("OT : Precision: %.4f; Recall: %.4f; F1: %.4f" % (
            eval_reporter['OT_p'], eval_reporter['OT_r'], eval_reporter['OT_f']))
        print("OW : Precision: %.4f; Recall: %.4f; F1: %.4f" % (
            eval_reporter['OW_p'], eval_reporter['OW_r'], eval_reporter['OW_f']))
        print("Relation : Precision: %.4f; Recall: %.4f; F1: %.4f ; ovrlap : %.4f" % (
            eval_triple_p, eval_triple_r, eval_triple_f, ovrlap_recall))
        print('-' * 12)


def init_model_and_data(opt):
    model = NER_NET(opt)
    train_iter, len_dataset = create_batch_iter('train', opt)
    test_iter, _ = create_batch_iter('test', opt)
    return model, train_iter, test_iter, len_dataset


def main(opt):
    start_time = time.time()
    model, train_iter, test_iter, len_dataset = init_model_and_data(opt)
    model = model.to(opt.device)
    model.load_state_dict(torch.load(common.output_dir + '/model.pt'))
    load_data_time = time_since(start_time)
    print('Time for loading the data: %.1f' % load_data_time)
    start_time = time.time()

    optimizer = AdamW(model.parameters(), lr=opt.learning_rate,
                      correct_bias=False)  # To reproduce BertAdam specific behavior set correct_bias=False
    scheduler = get_linear_schedule_with_warmup(optimizer,
                                                num_warmup_steps=int(
                                                    opt.warmup_proportion * len_dataset // opt.train_batch_size * opt.num_epoch),
                                                num_training_steps=len_dataset // opt.train_batch_size * opt.num_epoch)  # PyTorch scheduler

    train_model(model, optimizer, scheduler, train_iter, test_iter, opt, len_dataset)
    training_time = time_since(start_time)
    print('Time for training: %.1f' % training_time)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='train.py',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('-pos_embed_size', type=int, default=64,
                        help="Size of the pos types")
    parser.add_argument('-label_embed_size', type=int, default=32,
                        help="Size of the bio labels")
    parser.add_argument('-train_batch_size', type=int, default=32,
                        help="Size of the bio labels")
    parser.add_argument('-test_batch_size', type=int, default=32,
                        help="Size of the bio labels")
    parser.add_argument('-pos_after_gcn', action='store_true', default=True,
                        help="Size of the source vocabulary")
    parser.add_argument('-num_epoch', type=int, default=100,
                        help="Size of the source vocabulary")
    parser.add_argument('-patience', type=int, default=10,
                        help="Size of the source vocabulary")
    parser.add_argument('-hidden_size', type=int, default=768,
                        help="Size of the source vocabulary")
    parser.add_argument('-gcn_layer', type=int, default=1,
                        help="Size of the source vocabulary")
    parser.add_argument('-gcn_dropout', type=float, default=0.5,
                        help="Size of the source vocabulary")
    parser.add_argument('-warmup_proportion', type=float, default=0.1,
                        help="Size of the source vocabulary")
    parser.add_argument('-dropout', type=float, default=0.5,
                        help="Size of the source vocabulary")
    parser.add_argument('-learning_rate', type=float, default=1e-4,
                        help="Size of the source vocabulary")
    parser.add_argument('-rel_loss_weight', type=float, default=10000,
                        help="Size of the source vocabulary")
    parser.add_argument('-print_every', type=float, default=10,
                        help="Size of the source vocabulary")
    parser.add_argument('-device', type=str, default='cuda',
                        help="Size of the source vocabulary")

    opt = parser.parse_args()
    main(opt)
