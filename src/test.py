import pandas as pd
from pymongo import MongoClient
from src.model.svm_model import SVMModel
import random

client = MongoClient()


class TextClassificationPredict(object):
    def __init__(self) -> object:
        self.test = None

    def get_train_data(self):
        while True:
            # connect database
            client = MongoClient("mongodb://localhost:27017/")
            mydatabase = client.chatbot
            # mycollection = mydatabase.questiondata
            conn = MongoClient()
            db = conn.chatbot
            collectiondatatrain = db.questiondata
            collectionanswerdata = db.answerdata

            # To find() all the entries inside collection name 'myTable'
            cursor = collectiondatatrain.find()
            df_train = pd.DataFrame(cursor)
            test_data = []
            test = input('Bạn     => ')
            test_data.append({"feature": test, "target": "chao_hoi"})
            df_test = pd.DataFrame(test_data)
            # init model naive bayes
            model = SVMModel()
            clf = model.clf.fit(df_train["feature"], df_train.target)
            predicted1 = clf.predict(df_test["feature"])

            predict_p = clf.predict_proba(df_test["feature"])
            index = [i for i, r in enumerate(predict_p) if (r > 0.1).any()]
            targets = [df_test["feature"][i] for i in index]
            if (len(targets) > 0):
                predicted = clf.predict(df_test["feature"])
            else:
                predicted = 'khong_hieu'

            cursor1 = collectionanswerdata.find()
            customer = [customers for index, customers in enumerate(cursor1) if customers.get('target') == predicted1]
            if len(customer) > 0:
                feature = customer[0]["feature"]
                print("Chatbot => " + random.choice(feature))
                if predicted == 'ket_thuc':
                    return False


if __name__ == '__main__':
    tcp = TextClassificationPredict()
    tcp.get_train_data()
