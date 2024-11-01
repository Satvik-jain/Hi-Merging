import os
import pandas as pd
from multiprocessing import Pool
from tqdm import tqdm

# 全局变量声明
question_dict = None
answer_dict = None


# 初始化函数：用于在每个子进程中初始化全局变量
def init_worker(q_dict, a_dict):
    global question_dict, answer_dict
    question_dict = q_dict
    answer_dict = a_dict


# 定义训练集的替换函数
def replace_train_row(row):
    question_id, pos_ans_id = row[0], row[1]
    question = question_dict[question_id]
    pos_answer = answer_dict[pos_ans_id]
    return {
        "question": question,
        "pos_ans": pos_answer,
    }


# 定义验证集和测试集的替换函数
def replace_candidate_row(row):
    question_id, ans_id, cnt, label = row[0], row[1], row[2], row[3]
    question = question_dict[question_id]
    answer = answer_dict[ans_id]
    return {
        "question": question,
        "ans": answer,
        "cnt": cnt,
        "label": label
    }


def process_train_parallel(train_data, n_jobs=8):
    with Pool(processes=n_jobs, initializer=init_worker, initargs=(question_dict, answer_dict)) as pool:
        results = list(tqdm(pool.imap(replace_train_row, train_data.itertuples(index=False, name=None)), total=len(train_data)))
    return pd.DataFrame(results)


def process_candidates_parallel(candidate_data, n_jobs=8):
    with Pool(processes=n_jobs, initializer=init_worker, initargs=(question_dict, answer_dict)) as pool:
        results = list(tqdm(pool.imap(replace_candidate_row, candidate_data.itertuples(index=False, name=None)), total=len(candidate_data)))
    return pd.DataFrame(results)


if __name__ == '__main__':
    # 文件路径
    data_path = "./cMedQA2/"
    preprocessed_path = "./data/preprocessed/"
    dedup_neg_path = "./data/deduplicate_neg/"
    dedup_all_path = "./data/deduplicate_all/"

    # 检查并创建保存路径
    for path in [preprocessed_path, dedup_neg_path, dedup_all_path]:
        if not os.path.exists(path):
            os.makedirs(path)

    # 读取CSV文件
    questions = pd.read_csv(data_path + "question.csv")
    answers = pd.read_csv(data_path + "answer.csv")
    train_data = pd.read_csv(data_path + "train_candidates.txt", sep=",")
    dev_data = pd.read_csv(data_path + "dev_candidates.txt", sep=",")
    test_data = pd.read_csv(data_path + "test_candidates.txt", sep=",")

    # 将 questions 和 answers 转为字典形式，方便快速查找
    question_dict = pd.Series(questions.content.values, index=questions.question_id).to_dict()
    answer_dict = pd.Series(answers.content.values, index=answers.ans_id).to_dict()

    """第一步：ID 替换为文本内容，保存到 preprocessed 目录"""
    print("Step 1: Processing IDs to content and saving to preprocessed")

    # 替换 train 的 id 为文本内容
    train_samples = process_train_parallel(train_data, n_jobs=16)
    dev_samples = process_candidates_parallel(dev_data, n_jobs=16)
    test_samples = process_candidates_parallel(test_data, n_jobs=16)

    # 保存为JSON文件
    train_samples.to_json(preprocessed_path + "train.json", orient='records')
    dev_samples.to_json(preprocessed_path + "validation.json", orient='records')
    test_samples.to_json(preprocessed_path + "test.json", orient='records')

    """第二步：去除负样本，保存到 deduplicate_neg 目录"""
    print("Step 2: Removing negative samples and saving to deduplicate_neg")

    # 删除 train_data 的 neg_ans 列并去重
    train_data_cleaned = train_data.drop(columns=['neg_ans_id']).drop_duplicates(subset=['question_id', 'pos_ans_id'])

    # 删除 dev_data 和 test_data 中 label=0 的负样本
    dev_data_cleaned = dev_data[dev_data['label'] == 1]
    test_data_cleaned = test_data[test_data['label'] == 1]

    # 处理 train、validation 和 test
    train_samples_dedup_neg = process_train_parallel(train_data_cleaned, n_jobs=16)
    dev_samples_dedup_neg = process_candidates_parallel(dev_data_cleaned, n_jobs=16)
    test_samples_dedup_neg = process_candidates_parallel(test_data_cleaned, n_jobs=16)

    # 删除 dev_data 和 test_data 中 cnt 和 label 列
    dev_samples_dedup_neg = dev_samples_dedup_neg.drop(columns=['cnt', 'label'])
    test_samples_dedup_neg = test_samples_dedup_neg.drop(columns=['cnt', 'label'])

    # 将字段名改为 question、answer
    train_samples_dedup_neg = train_samples_dedup_neg.rename(columns={"pos_ans": "answer"})
    dev_samples_dedup_neg = dev_samples_dedup_neg.rename(columns={"ans": "answer"})
    test_samples_dedup_neg = test_samples_dedup_neg.rename(columns={"ans": "answer"})

    # 保存为JSON文件
    train_samples_dedup_neg.to_json(dedup_neg_path + "train.json", orient='records')
    dev_samples_dedup_neg.to_json(dedup_neg_path + "validation.json", orient='records')
    test_samples_dedup_neg.to_json(dedup_neg_path + "test.json", orient='records')

    """第三步：去除重复问题和答案，保存到 deduplicate_all 目录"""
    print("Step 3: Removing duplicate answers and saving to deduplicate_all")

    # 删除 train 中重复的问题，只保留第一个正样本
    train_data_final = train_data_cleaned.drop_duplicates(subset=['question_id'])

    # 删除 dev 和 test 中的重复答案，只保留第一个
    dev_data_final = dev_data_cleaned.drop_duplicates(subset=['question_id'])
    test_data_final = test_data_cleaned.drop_duplicates(subset=['question_id'])

    # 处理 train、validation 和 test
    train_samples_dedup_all = process_train_parallel(train_data_final, n_jobs=16)
    dev_samples_dedup_all = process_candidates_parallel(dev_data_final, n_jobs=16)
    test_samples_dedup_all = process_candidates_parallel(test_data_final, n_jobs=16)

    # 删除 dev_data 和 test_data 中 cnt 和 label 列
    dev_samples_dedup_all = dev_samples_dedup_all.drop(columns=['cnt', 'label'])
    test_samples_dedup_all = test_samples_dedup_all.drop(columns=['cnt', 'label'])

    # 将字段名改为 question、answer
    train_samples_dedup_all = train_samples_dedup_all.rename(columns={"pos_ans": "answer"})
    dev_samples_dedup_all = dev_samples_dedup_all.rename(columns={"ans": "answer"})
    test_samples_dedup_all = test_samples_dedup_all.rename(columns={"ans": "answer"})

    # 保存为JSON文件
    train_samples_dedup_all.to_json(dedup_all_path + "train.json", orient='records')
    dev_samples_dedup_all.to_json(dedup_all_path + "validation.json", orient='records')
    test_samples_dedup_all.to_json(dedup_all_path + "test.json", orient='records')

    print("Data processing complete.")

