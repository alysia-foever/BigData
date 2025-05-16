import re
import jieba
from tqdm import tqdm
import numpy as np
from gensim.models import Word2Vec
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, classification_report
from imblearn.over_sampling import RandomOverSampler
from sklearn.model_selection import GridSearchCV

# 读取数据并划分标签和文本
def divide_dataset(filename, lines=1000):
    with open(filename, 'r', encoding='utf-8') as f:
        text_data = f.readlines()

    # 选择前lines行的数据
    subset = text_data[:lines]
    # 分离每一行的标签和文本
    dataset = [s.strip().split('\t') for s in subset]
    
    # 去除空文本项
    dataset = [data for data in dataset if len(data) == 2 and data[1].strip()]
    
    tag = [data[0] for data in dataset]
    text = [data[1] for data in dataset]

    return tag, text

# 文本清洗
def clean_text(dataset):
    cleaned_text = []
    # for text in tqdm(dataset, desc='Cleaning text'):
    for text in dataset:
        # 仅保留中文字符、字母和数字
        clean = re.sub(r'[^\u4e00-\u9fa5a-zA-Z0-9\s]', '', text)  
        # 处理缺失值和异常值
        cleaned_text.append(clean.strip())
    return cleaned_text

# 文本标记化和停用词处理
def tokenize_and_remove_stopwords(dataset):
    stopwords_file = '第四章/基本示例/数据集/hit_stopwords.txt'
    with open(stopwords_file, 'r', encoding='utf-8') as file:
        stopwords = {line.strip() for line in file}

    tokenized_text = []
    # for text in tqdm(dataset, desc='Tokenizing and removing stopwords'):
    for text in dataset:
        # 使用jieba进行分词
        words = jieba.lcut(text)  
        # 移除停用词
        filtered_words = [word for word in words if word not in stopwords]  
        tokenized_text.append(filtered_words)

    return tokenized_text

# 特征提取
def generate_text_vectors(tokenized_text):
    model = Word2Vec(sentences=tokenized_text, vector_size=100, window=5, min_count=1, sg=0)
    word_vectors = model.wv

    text_vectors = []
    # for tokens in tqdm(tokenized_text, desc='Generating text vectors'):
    for tokens in tokenized_text:
        # 转换为词向量表示
        vectors = [word_vectors[word] for word in tokens if word in word_vectors]  
        if vectors:
            # 取平均值
            text_vectors.append(np.mean(vectors, axis=0))  
        else:
            # 如果没有词向量则用0向量代替
            text_vectors.append(np.zeros(100))  

    return text_vectors

# 垃圾文本分类
def spam_classification(train_tags, train_word_vectors, test_tags, test_word_vectors):
    # 使用RandomOverSampler进行过采样
    #oversampler = RandomOverSampler(sampling_strategy=0.5, random_state=42)
    #X_resampled, y_resampled = oversampler.fit_resample(train_word_vectors, train_tags)

    # 使用支持向量机分类器
    svm_classifier = SVC(kernel='linear')

    # 定义参数网格
    #param_grid = {
    #    'kernel': ['linear', 'rbf'],  # 选择核函数
    #    'C': [0.1, 0.5, 1, 5, 10],             # 正则化参数
    #}
    # 创建GridSearchCV对象
    #grid_search = GridSearchCV(svm_classifier, param_grid, cv=5, scoring='recall', verbose=2, n_jobs=-1)
    # 在过采样后的训练数据上进行网格搜索
    #grid_search.fit(train_word_vectors, train_tags)
    # 输出最佳参数
    #print("最佳参数组合:", grid_search.best_params_)

    svm_classifier.fit(np.array(train_word_vectors), np.array(train_tags))  

    # 在测试集上进行预测并显示进度条
    predictions = []
    for vector in tqdm(test_word_vectors, desc='Classifying', leave=False):
        prediction = svm_classifier.predict([vector])
        #prediction = grid_search.predict([vector])
        predictions.append(prediction[0])

    # 输出混淆矩阵和分类报告
    cm = confusion_matrix(np.array(test_tags), np.array(predictions))
    print("混淆矩阵:")
    print(cm)

    report = classification_report(np.array(test_tags), np.array(predictions))
    print("分类报告:")
    print(report)

    # 输出模型评估结果
    # accuracy = accuracy_score(np.array(test_tags), np.array(predictions))
    # print(f'准确率: {accuracy:.2f}')

if __name__ == "__main__":
    train_tags, train_text = divide_dataset("第四章/基本示例/数据集/train.txt", 100000)
    test_tags, test_text = divide_dataset("第四章/基本示例/数据集/test.txt", 200000)

    cleaned_train_text = clean_text(train_text)
    cleaned_test_text = clean_text(test_text)

    train_tokenized_text = tokenize_and_remove_stopwords(cleaned_train_text)
    test_tokenized_text = tokenize_and_remove_stopwords(cleaned_test_text)

    train_word_vectors = generate_text_vectors(train_tokenized_text)
    test_word_vectors = generate_text_vectors(test_tokenized_text)

    spam_classification(train_tags, train_word_vectors, test_tags, test_word_vectors)