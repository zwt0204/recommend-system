# -*- encoding: utf-8 -*-
"""
@File    : UserCF.py
@Time    : 2020/6/1 17:55
@Author  : zwt
@git   : https://github.com/xingzhexiaozhu/MovieRecommendation
@Software: PyCharm
"""
import random
import math
from operator import itemgetter


class UserBasedCF(object):

    def __init__(self):
        self.n_sim_user = 20  # 与目标用户兴趣相似的20个用户
        self.n_rec_movie = 10   # 推荐10部电影

        # 将数据集划分为训练集和测试集
        self.trainSet = {}
        self.testSet = {}

        # 用户相似度矩阵
        self.user_sim_matrix = {}
        self.temp = {}
        self.movie_count = 0

    def get_dataset(self, filename, pivot=0.75):
        """
        读文件得到“用户-电影”数据
        :param filename: 文件路径
        :param pivot: 训练测试集分割阈值
        :return: 
        """
        train_set_len = 0
        test_set_len = 0
        for line in self.load_file(filename):
            user, movie, rating, timestamp = line.split(',')
            if random.random() < pivot:
                self.trainSet.setdefault(user, {})
                self.trainSet[user][movie] = rating
                train_set_len += 1
            else:
                self.testSet.setdefault(user, {})
                self.testSet[user][movie] = rating
                test_set_len += 1
        print('Split trainingSet and testSet success!')
        print('TrainSet = %s' % train_set_len)
        print('TestSet = %s' % test_set_len)

    @staticmethod
    def load_file(filename):
        """
        读文件，返回文件的每一行
        :param filename:
        :return:
        """
        with open(filename, 'r') as f:
            for i, line in enumerate(f):
                if i == 0:
                    continue
                yield line.strip('\r\n')
        print('Load %s success!' % filename)

    # 计算用户之间的相似度
    def calc_user_sim(self):
        # 构建“电影-用户”倒排索引
        # key = movieID, value = list of userIDs who have seen this movie
        print('Building movie-user table ...')
        movie_user = {}
        for user, movies in self.trainSet.items():
            for movie in movies:
                if movie not in movie_user:
                    movie_user[movie] = set()
                movie_user[movie].add(user)
        print('Build movie-user table success!')

        self.movie_count = len(movie_user)
        print('Total movie number = %d' % self.movie_count)

        print('===temp===start')
        for movie, users in movie_user.items():
            for u in users:
                for v in users:
                    if u == v:
                        continue
                    self.temp.setdefault(u, {})
                    self.temp[u].setdefault(v, set())
                    self.temp[u][v].add(movie)
        print('===temp===end')

        print('Build user co-rated movies matrix ...')
        for movie, users in movie_user.items():
            for u in users:
                for v in users:
                    if u == v:
                        continue
                    self.user_sim_matrix.setdefault(u, {})
                    self.user_sim_matrix[u].setdefault(v, 0)
                    self.user_sim_matrix[u][v] += 1
        print('Build user co-rated movies matrix success!')
        # print(self.user_sim_matrix)
        # 计算相似性
        print('Calculating user similarity matrix ...')
        # 原始的相似度计算：precisioin=0.3020	recall=0.0729	coverage=0.0422
        # for u, related_users in self.user_sim_matrix.items():
        #     for v, count in related_users.items():
        #         self.user_sim_matrix[u][v] = count / math.sqrt(len(self.trainSet[u]) * len(self.trainSet[v]))

        # 改进的相似度计算：precisioin=0.3011	recall=0.0732	coverage=0.0475
        for u, related_items in self.temp.items():
            for v, related in related_items.items():
                sum = 0
                for i in related:
                    sum += 1 / math.log(1+len(movie_user[i]), 10)
                self.user_sim_matrix[u][v] = sum / math.sqrt(len(self.trainSet[u]) * len(self.trainSet[v]))

        print('Calculate user similarity matrix success!')

    # 针对目标用户U，找到其最相似的K个用户，产生N个推荐
    def recommend(self, user):
        K = self.n_sim_user
        N = self.n_rec_movie
        rank = {}
        watched_movies = self.trainSet[user]
        # v=similar user, wuv=similar factor
        for v, wuv in sorted(self.user_sim_matrix[user].items(), key=itemgetter(1), reverse=True)[0:K]:
            for movie in self.trainSet[v]:
                if movie in watched_movies:
                    continue
                rank.setdefault(movie, 0)
                rank[movie] += wuv
        return sorted(rank.items(), key=itemgetter(1), reverse=True)[0:N]

    # 产生推荐并通过准确率、召回率和覆盖率进行评估
    def evaluate(self):
        print("Evaluation start ...")
        N = self.n_rec_movie
        # 准确率和召回率
        hit = 0
        rec_count = 0
        test_count = 0
        # 覆盖率
        all_rec_movies = set()

        for i, user, in enumerate(self.trainSet):
            test_movies = self.testSet.get(user, {})
            rec_movies = self.recommend(user)
            for movie, w in rec_movies:
                if movie in test_movies:
                    hit += 1
                all_rec_movies.add(movie)
            rec_count += N
            test_count += len(test_movies)

        precision = hit / (1.0 * rec_count)
        recall = hit / (1.0 * test_count)
        coverage = len(all_rec_movies) / (1.0 * self.movie_count)
        print('precisioin=%.4f\trecall=%.4f\tcoverage=%.4f' % (precision, recall, coverage))


if __name__ == '__main__':
    rating_file = 'D:\迅雷下载\ml-latest-small\ml-latest-small\\ratings.csv'
    userCF = UserBasedCF()
    userCF.get_dataset(rating_file)
    userCF.calc_user_sim()
    userCF.evaluate()
