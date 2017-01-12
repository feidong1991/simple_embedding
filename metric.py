#!/usr/bin/env python

# coding=utf-8



##===================================================================##

#   Utils:load data

#   Jie Yang

#   Sep. 7, 2016

# 

##===================================================================##



# from operator import add

# 

import numpy as np

import math

def get_golden_predict_choose_results(structured_sentences, predict_result):

	whole_index = 0

	right_num = 0

	golden_choose_list = []

	predict_choose_list = []

	for each_candidates in structured_sentences:

		candidate_index = 0

		predict_candidate_score = -1

		predict_opt_candidate = -1

		golden_candidate_score = -1

		golden_opt_candidate = -1

		for each_sentence in each_candidates:

			if (predict_result[whole_index] > predict_candidate_score):

				predict_candidate_score = predict_result[whole_index]

				predict_opt_candidate = candidate_index

			if each_sentence[5] > golden_candidate_score:

				golden_candidate_score = each_sentence[5]

				golden_opt_candidate = candidate_index

			candidate_index += 1

			whole_index += 1

		golden_choose_list.append(golden_opt_candidate)

		predict_choose_list.append(predict_opt_candidate)

	return golden_choose_list,predict_choose_list





def get_alpha_golden_predict_choose_results(structured_sentences, predict_result,alpha):

    whole_index = 0

    right_num = 0

    golden_choose_list = []

    predict_choose_list = []

    for each_candidates in structured_sentences:

        candidate_index = 0

        predict_candidate_score = -1

        predict_opt_candidate = -1

        golden_candidate_score = -1

        golden_opt_candidate = -1

        # predict_index_sum = 0.0

        # temp_add_index = 0

        # for each_sentence in each_candidates:

        #     predict_index_sum += math.exp(predict_result[whole_index+temp_add_index])

        #     temp_add_index += 1

        for each_sentence in each_candidates:

            predict_value = predict_result[whole_index] * alpha + (1-alpha)* each_sentence[7]

            #predict_value = predict_result[whole_index] * each_sentence[7]

            # predict_value = math.exp(predict_result[whole_index]) * each_sentence[7]/predict_index_sum

            if (predict_value > predict_candidate_score):

                predict_candidate_score = predict_value

                predict_opt_candidate = candidate_index

            if each_sentence[5] > golden_candidate_score:

                golden_candidate_score = each_sentence[5]

                golden_opt_candidate = candidate_index

            candidate_index += 1

            whole_index += 1

        golden_choose_list.append(golden_opt_candidate)

        predict_choose_list.append(predict_opt_candidate)

    return golden_choose_list,predict_choose_list





def candidate_choose_accuracy(golden_list, predict_list):

    result_num = len(golden_list)

    same_number = 0

    assert (result_num == len(predict_list)),"Golden and predict result size not match!"

    for idx in range(0,result_num):

      if golden_list[idx] == predict_list[idx]:

          same_number += 1

    accuracy = (same_number+0.0)/result_num

    # print "Total instances: ", result_num, "; Correct choice: ", same_number, "; Accuracy: ",accuracy

    return accuracy







def get_rerank_ner_fmeasure(structured_sentences, predict_choose_list):

	seq_num = len(predict_choose_list)

	assert(len(structured_sentences) == seq_num), "structured_sentence and predict choose num not match! sent_Num:predict"

	sentence_list = []

	golden_list = []

	predict_list = []

	for idx in range(0, seq_num):

		sentence_list.append(structured_sentences[idx][predict_choose_list[idx]][2])

		golden_list.append(structured_sentences[idx][predict_choose_list[idx]][3])

		predict_list.append(structured_sentences[idx][predict_choose_list[idx]][4])

	return get_ner_fmeasure(sentence_list,golden_list,predict_list)





def get_ner_fmeasure(golden_lists, predict_lists, label_type="BMES"):

    sent_num = len(golden_lists)

    # assert(sent_num == len(golden_lists)), "Sentence num and golden num not match!"

    golden_full = []

    predict_full = []

    right_full = []

    for idx in range(0,sent_num):

        # word_list = sentence_lists[idx]

        golden_list = golden_lists[idx]

        predict_list = predict_lists[idx]

        if label_type == "BMES":

            gold_matrix = get_ner_BMES(golden_list)

            pred_matrix = get_ner_BMES(predict_list)

        # else:

        #     gold_matrix = get_ner(word_list, golden_list)

        #     pred_matrix = get_ner(word_list, predict_list)

        right_ner = list(set(gold_matrix).intersection(set(pred_matrix)))

        golden_full += gold_matrix

        predict_full += pred_matrix

        right_full += right_ner

    right_num = len(right_full)

    golden_num = len(golden_full)

    predict_num = len(predict_full)

    precision =  (right_num+0.0)/predict_num

    recall = (right_num+0.0)/golden_num

    f_measure = 2*precision*recall/(precision+recall)

    # print "gold_num = ", golden_num, " pred_num = ", predict_num, " right_num = ", right_num

    return precision, recall, f_measure





def reverse_style(input_string):

    target_position = input_string.index('[')

    input_len = len(input_string)

    output_string = input_string[target_position:input_len] + input_string[0:target_position]

    return output_string







# def get_ner(word_list, label_list):

#     list_len = len(word_list)

#     assert(list_len == len(label_list)), "word list size unmatch with label list"

#     begin_label = 'B-'

#     inside_label = 'I-' 

#     # single_label = 'S-'

#     whole_tag = ''

#     index_tag = ''

#     tag_list = []

#     stand_matrix = []

#     for i in range(0, list_len):

#         wordlabel = word_list[i]

#         current_label = label_list[i].upper()

#         if begin_label in current_label:

#             if index_tag == '':

#                 whole_tag = current_label.strip(begin_label) +'[' +str(i)

#                 index_tag = current_label.strip(begin_label)

#             else:

#                 tag_list.append(whole_tag + ',' + str(i-1))

#                 whole_tag = current_label.strip(begin_label)  + '[' + str(i)

#                 index_tag = current_label.strip(begin_label)

#         elif 

#         elif inside_label in current_label:

#             if current_label.strip(inside_label) == index_tag:

#                 whole_tag = whole_tag 

#             else:

#                 if (whole_tag != '')&(index_tag != ''):

#                     tag_list.append(whole_tag +',' + str(i-1))

#                 whole_tag = ''

#                 index_tag = ''

#         else:

#             if (whole_tag != '')&(index_tag != ''):

#                 tag_list.append(whole_tag +',' + str(i-1))

#             whole_tag = ''

#             index_tag = ''

#     if (whole_tag != '')&(index_tag != ''):

#         tag_list.append(whole_tag)

#     tag_list_len = len(tag_list)

#     for i in range(0, tag_list_len):

#         if  len(tag_list[i]) > 0:

#             tag_list[i] = tag_list[i]+ ']'

#             insert_list = reverse_style(tag_list[i])

#             stand_matrix.append(insert_list)

#     return stand_matrix



def get_ner_BMES(label_list):

    # list_len = len(word_list)

    # assert(list_len == len(label_list)), "word list size unmatch with label list"

    list_len = len(label_list)

    begin_label = 'B-'

    inside_label = 'M-' 

    end_label = 'E-'

    single_label = 'S-'

    whole_tag = ''

    index_tag = ''

    tag_list = []

    stand_matrix = []

    for i in range(0, list_len):

        # wordlabel = word_list[i]

        current_label = label_list[i].upper()

        if begin_label in current_label:

            if index_tag == '':

                whole_tag = current_label.replace(begin_label,"",1) +'[' +str(i)

                index_tag = current_label.replace(begin_label,"",1)

            else:

                tag_list.append(whole_tag + ',' + str(i-1))

                whole_tag = current_label.replace(begin_label,"",1)  + '[' + str(i)

                index_tag = current_label.replace(begin_label,"",1)

        elif single_label in current_label:

            if index_tag != '':

                tag_list.append(whole_tag + ',' + str(i-1))

            whole_tag = current_label.replace(single_label,"",1) +'[' +str(i)

            tag_list.append(whole_tag)

            whole_tag = ""

            index_tag = ""



        elif inside_label in current_label:

            if current_label.replace(inside_label,"",1) == index_tag:

                whole_tag = whole_tag 

            else:

                if (whole_tag != '')&(index_tag != ''):

                    tag_list.append(whole_tag +',' + str(i-1))

                whole_tag = ''

                index_tag = ''

        elif end_label in current_label:

            if current_label.replace(end_label,"",1) == index_tag:

                tag_list.append(whole_tag +',' + str(i))

            whole_tag = ''

            index_tag = ''

        else:

            if (whole_tag != '')&(index_tag != ''):

                tag_list.append(whole_tag +',' + str(i-1))

            whole_tag = ''

            index_tag = ''



    if (whole_tag != '')&(index_tag != ''):

        tag_list.append(whole_tag)

    tag_list_len = len(tag_list)



    for i in range(0, tag_list_len):

        if  len(tag_list[i]) > 0:

            tag_list[i] = tag_list[i]+ ']'

            insert_list = reverse_style(tag_list[i])

            stand_matrix.append(insert_list)

    # print stand_matrix

    return stand_matrix







def readSentence(input_file):

    in_lines = open(input_file,'r').readlines()

    sentences = []

    labels = []

    sentence = []

    label = []

    for line in in_lines:

        if len(line) < 2:

            sentences.append(sentence)

            labels.append(label)

            sentence = []

            label = []

        else:

            pair = line.strip('\n').split(' ')

            sentence.append(pair[0])

            label.append(pair[-1])

    return sentences,labels







# def isStartLabel(label):

#     if "B-" in label.upper():

#         return True

#     elif "S-" in label.upper():

#         return True

#     else:

#         return False



# def isContinueLabel(label):

#     if 

if __name__ == '__main__':

    input_file = "seg.test.pd"

    sentences,labels = readSentence(input_file)

    get_ner_BMES(labels[0])



