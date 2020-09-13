from sklearn.metrics import f1_score, accuracy_score, classification_report
import common
import torch
import numpy as np

BIO_tag_map = common.labelDic


def flatlist(listlist):
    result = []
    for l in listlist:
        result.extend(l)
    return result


def get_tags_BIO(path, tag, tag_map=BIO_tag_map):
    b_tag = tag_map.get("B-" + tag)
    i_tag = tag_map.get("I-" + tag)
    o_tag = tag_map.get("O")
    path = [p if p in [b_tag, i_tag] else 0 for p in path]
    mentions = []
    c_start = -1
    last_tag = o_tag
    for idx, tag in enumerate(path):
        if last_tag == o_tag:
            if tag == b_tag:
                c_start = idx
        elif last_tag == i_tag and c_start != -1:
            if tag == b_tag:
                mentions.append([c_start, idx - 1])
                c_start = idx
            elif tag == o_tag:
                mentions.append([c_start, idx - 1])
                c_start = -1
        elif last_tag == b_tag:
            if tag == b_tag and c_start != -1:
                mentions.append([c_start, idx - 1])
                c_start = idx
            elif tag == o_tag and c_start != -1:
                mentions.append([c_start, idx - 1])
                c_start = -1
        last_tag = tag
    if c_start != -1:
        mentions.append([c_start, len(path) - 1])
    return mentions


if __name__ == '__main__':
    path = [0, 1, 2, 3, 4, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3, 0]
    tags = get_tags_BIO(path, 'P')
    print(tags)


def eval_entity(tar_path, pre_path, tag):
    origin = 0.
    found = 0.
    right = 0.
    for fetch in zip(tar_path, pre_path):
        tar, pre = fetch
        tar_tags = get_tags_BIO(tar, tag, BIO_tag_map)
        pre_tags = get_tags_BIO(pre, tag, BIO_tag_map)
        origin += len(tar_tags)
        found += len(pre_tags)

        for p_tag in pre_tags:
            if p_tag in tar_tags:
                right += 1
    recall = 0. if origin == 0 else (right / origin)
    precision = 0. if found == 0 else (right / found)
    f1 = 0. if recall + precision == 0 else (2 * precision * recall) / (precision + recall)
    return precision, recall, f1


def eval_result(predicts, labels, show=False):
    '''
    :param predicts: list[list,]
    :param labels: tensor pad by -1
    :return:
    '''
    label_list = []
    for label in labels:
        label = label[label != -1].numpy().tolist()
        label_list.append(label)
    pre_flat = flatlist(predicts)
    lab_flat = flatlist(label_list)

    if show:
        cr = classification_report(lab_flat, pre_flat)
        print(cr)

    tag_acc = accuracy_score(lab_flat, pre_flat)
    tag_f = f1_score(lab_flat, pre_flat, average='macro')

    fusion_p, fusion_r, fusion_f = eval_entity(label_list, predicts, 'T')
    part_p, part_r, part_f = eval_entity(label_list, predicts, 'P')

    reporter = {'tag_acc': tag_acc,
                'tag_f': tag_f,
                'OT_p': fusion_p,
                'OT_r': fusion_r,
                'OT_f': fusion_f,
                'OW_p': part_p,
                'OW_r': part_r,
                'OW_f': part_f}
    return reporter


def make_pre_relation(pre_rel, eval_predicted_label, thred=0.5):
    total_pre_result = []
    for pr, predict_label in zip(pre_rel, eval_predicted_label):
        pr = pr[:, :, 1]
        targetList = get_tags_BIO(predict_label, tag='T')
        opinionList = get_tags_BIO(predict_label, tag='P')
        pre_relationResult = []
        for o in opinionList:
            for t in targetList:
                score = (np.sum(pr[o[0]:o[1] + 1, t[0]:t[1] + 1]) + np.sum(pr[t[0]:t[1] + 1, o[0]:o[1] + 1])) / (
                        (o[1] + 1 - o[0]) * (t[1] + 1 - t[0]) * 2)
                if score > thred:
                    if [o[0], o[1], t[0], t[1]] not in pre_relationResult:
                        pre_relationResult.append([o[0], o[1], t[0], t[1]])

        total_pre_result.append(pre_relationResult)
        check = [' '.join([str(t) for t in tt]) for tt in pre_relationResult]
        if len(check) != len(set(check)):
            print('error', total_pre_result)
    return total_pre_result


def make_gold_relation(tar_rel, label_id):
    label_list = []
    for label in label_id:
        label = label[label != -1].numpy().tolist()
        label_list.append(label)

    total_rel_result = []
    for relation, label in zip(tar_rel, label_list):
        targetList = get_tags_BIO(label, tag='T')
        opinionList = get_tags_BIO(label, tag='P')
        relationResult = []
        for o in opinionList:
            for t in targetList:
                if np.sum(relation[o[0]:o[1] + 1, t[0]:t[1] + 1]) == (o[1] + 1 - o[0]) * (t[1] + 1 - t[0]):
                    relationResult.append([o[0], o[1], t[0], t[1]])
        total_rel_result.append(relationResult)
    return total_rel_result


def eval_rel_by_condition(pre_rel, eval_predicted_label, tar_rel, label_id, thred=0.5):
    '''
    :param pre_rel: batch, seqlen, seqlen, 2 numpy
    :param tar_rel: batch, seqlen, seqlen,numpy
    :param output_mask: batch, seqlen
    :return:
    '''

    pre_relation_list = make_pre_relation(pre_rel, eval_predicted_label, thred=thred)
    tar_relation_list = make_gold_relation(tar_rel, label_id)

    goldTotal = 0
    correct = 0
    predictTotal = 0

    goldOvrlapTotal = 0
    correctOvrlap = 0

    for pred, standard in zip(pre_relation_list, tar_relation_list):
        overlap_relations = get_overlap_relation(standard)

        goldTotal += len(standard)
        predictTotal += len(pred)
        goldOvrlapTotal += len(overlap_relations)
        for r in pred:
            if r in standard:
                correct += 1
            if r in overlap_relations:
                correctOvrlap += 1

    precision = float(correct) / (predictTotal + 1e-6)
    recall = float(correct) / (goldTotal + 1e-6)
    f1 = 2 * precision * recall / (precision + recall + 1e-6)
    ovrlap_recall = float(correctOvrlap) / (goldOvrlapTotal + 1e-6)
    return precision, recall, f1, ovrlap_recall


def get_overlap_relation(relations):
    overlap_relations = []
    for idx1 in range(len(relations) - 1):
        for idx2 in range(idx1 + 1, len(relations)):
            relation1 = relations[idx1]
            relation2 = relations[idx2]
            if (relation1[0] == relation2[0] and relation1[1] == relation2[1]) or (
                    relation1[2] == relation2[3] and relation1[2] == relation2[3]):
                overlap_relations.append(relation1)
    return overlap_relations


def cont_overlap_relation(tar_rel, label_id):
    tar_relation = make_gold_relation(tar_rel, label_id)
    all_overlap = 0
    all_over_lap_relations = []
    for relations in tar_relation:
        over_lap_relations = []
        for idx1 in range(len(relations) - 1):
            for idx2 in range(idx1 + 1, len(relations)):
                relation1 = relations[idx1]
                relation2 = relations[idx2]
                if (relation1[0] == relation2[0] and relation1[1] == relation2[1]) or (
                        relation1[2] == relation2[3] and relation1[2] == relation2[3]):
                    all_overlap += 1
                    over_lap_relations.append(relation1)
        all_over_lap_relations.append(over_lap_relations)
    print('all overlap relation', all_overlap)
