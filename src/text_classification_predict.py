#!/usr/bin/env python
# -*- coding: utf-8 -*-
import pandas as pd
from src.model.svm_model import SVMModel
# from src.model.naive_bayes_model import NaiveBayesModel
from pymongo import MongoClient


class TextClassificationPredict(object):
    def __init__(self) -> object:
        self.test = None

    def get_train_data(self, msg):
        #  train data
        # while True:
        # init MongoDB
        myclient = MongoClient("mongodb://localhost:27017/")
        mydb = myclient.chatbot
        mycol = mydb.traindata
        conn = MongoClient()
        db = conn.chatbot
        collecttraindata = db.questiondata
        collectanswerdata = db.answerdata
        cursor = collecttraindata.find()
        df_train = pd.DataFrame(cursor)

        #  test train data
        test_data = []
        # test = input('Ban => ')
        print(msg)
        test = msg
        test_data.append({"feature": test, "target": "init"})
        df_test = pd.DataFrame(test_data)

        # init model naive bayes
        model = SVMModel()
        clf = model.clf.fit(df_train["feature"], df_train.target)
        score = clf.predict_proba(df_test["feature"])
        index = [i for i, r in enumerate(score) if (r > 0.1).any()]
        targets = [df_test["feature"][i] for i in index]
        if (len(targets) > 0):
            predicted = clf.predict(df_test["feature"])
        else:
            predicted = 'khong_hieu'
        #print(score)
        #print(predicted)
        # connect data_train file to answer_data file
        cursor1 = collectanswerdata.find()
        customer = [customers for index, customers in enumerate(cursor1) if customers.get('target') == predicted]
        try:
            if len(customer) > 0:
                feature = customer[0]["feature"]
                import random
                return (random.choice(feature))
                if predicted == 'ket_thuc':
                    return False
        except:
            return ("")

#if __name__ == '__main__':
 #tcp = TextClassificationPredict()
 #tcp.get_train_data("Con tôi có ý định học ngành công nghệ thông tin, vậy các ngành đào tạo công nghệ thông tin?")
