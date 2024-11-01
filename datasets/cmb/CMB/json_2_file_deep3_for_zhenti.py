import os
import json

# 读取JSON文件
json_file_path = 'C:/Users/xidongw/Desktop/examatlas/CMB/CMB-test-exampaper/CMB-test-exam-merge.json'
with open(json_file_path, 'r', encoding='utf-8') as f:
    data = json.load(f)

# 对数据进行分类
categories = {}
for item in data:
    exam_type = item.get('exam_type', 'other')
    exam_class = item.get('exam_class', 'other')
    exam_subject = item.get('exam_subject', 'other')

    if exam_type not in categories:
        categories[exam_type] = {}
    if exam_class not in categories[exam_type]:
        categories[exam_type][exam_class] = {}
    if exam_subject not in categories[exam_type][exam_class]:
        categories[exam_type][exam_class][exam_subject] = []

    categories[exam_type][exam_class][exam_subject].append(item)

# 将分类后的数据写入到相应的文件中
for exam_type, classes in categories.items():
    for exam_class, subjects in classes.items():
        for exam_subject, items in subjects.items():
            # 确保目录存在
            directory = os.path.join(os.path.dirname(json_file_path), exam_class)
            os.makedirs(directory, exist_ok=True)

            # 将数据写入到文件中
            file_path = os.path.join(directory, exam_subject + '.json')
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(items, f, ensure_ascii=False, indent=4)
