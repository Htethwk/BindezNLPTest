#!/usr/local/bin/python
# -*- coding: utf-8 -*-
import os,sys
from textblob.classifiers import NaiveBayesClassifier
from textblob import TextBlob
from nltk.corpus import indian
import glob

def read_lines(fname,type):
    comments = []
    owd = os.getcwd()
    os.chdir(fname)
    for file in glob.glob("*.txt"):
        f = open(file ,'r')
        lines = f.read()
        comments.append([lines.decode('utf8'),type]) #deconding and appending the type to each text
        f.close()
    os.chdir(owd)
    return comments

#test Cases
test = {("Ooredoo အခုတလော Bill တွေ ခိုး နေ တယ် နော်  .. Bill 1000 မ ဖြည့် ခင် ကစစ် လိုက် တာ 8 ကျပ် .. Bill ဖြည့် ပီး တော့ ပြန် စစ် လိုက် တာ 1004 ကျပ်   ပေါင်းကူး ဒေတာ 999 ကျပ် ချက်ချင်း ဝယ် ပီး စစ် လိုက် တာ 1 Kyats တဲ့   မင်း တို့ တော်တော် ကို တရားလွန် နေ ပီ .. ဖုန်းခေါ် ပီး ရင် လည်း ဒီ အတိုင်း ပဲ 1 ကျပ် 2 ကျပ် နဲ့ ခိုး ခိုး နေ တာ ,,, မင်း တို့ အဲ့ လို ကြပ်ကြပ် လုပ် မကြာခင် နာမည်ပျက် မှာ ပဲ    ငါ ကတော့ ဒီ တစိ ခါဘေလ် ကုန် ရင် Ooredoo မ သုံး တော့ ဘူး".decode('utf8'), 'Neg'),
         ("$ Ooredoo မ သုံး ကြ ပါ နဲ့ ။ ဘေ ခိုး တယ် ။ မ ယူ ချင် ပဲ နဲ့ Gameloft plan ယူ ခိုင်း တယ် ။ နည်း အမျိုးမျိုး နဲ့ ခိုး နေ တာ ".decode('utf8') , 'Neg'),
         ("6GB ရ ဖို့ နေနေသာသာ ဘာ လိုင်း မှ မ ပွင့် ဘူး".decode('utf8') , 'Neg'),
         ("ကျေးဇူး ပါ ဗျာ ဒေါင်း ပါပြီ ".decode('utf8'), 'Pos' ),
         ("aww အဲ့ လိုလား ကျေးဇူး ပါ သန်လျင် မှာ လိုင်း ပြန် ကောင် လာ တယ် Ooredoo ဒီ့ထက် မက အောင်မြင်ပါစေ ".decode('utf8') , 'Pos'),
         ("package ဝယ်သုံး တာ ".decode('utf8'),'Pos'),
         ("လစဉ် 3000 နဲ oore အချင်းချင်း plan ဘယ် လို့ လုပ် ရ သလဲ ".decode('utf8'), 'Question'),
         ("iflix ဝယ် ထား တာ ဘယ်လို သုံး ရတာ လဲ facebook လဲ သုံး လို့ မ ရ ဘူး ".decode('utf8'),'Question'),
         ("အော်ပရေတာ နဲ့ ဆက်သွယ် ရ မှာ လား ခ န လောက် သီး ခံ ပေး ပါ ....  ခ န လောက် သီး ခံ ပေး ပါ နဲ့ .....     သူ ကို ခေါ် ရတာ နဲ့ တင် တစ်ရက် တစ်ရက် မ လောက် ဘူးနေ့ မကောင်း ည မကောင်း တဲ့ စောင့်သုံး ရ တယ် ... ရိ့ လို ပဲ စိတ် ဆင် ရဲ တယ် ထုတ် ပြီ ဒါပဲ .....    Go away Ooredoo Sim Cards ရေ ".decode('utf8'),'Neg')

}

#Test Data for each set
pos_sentiment = read_lines("Positive",'Pos')
neg_sentiment = read_lines("Negative", 'Neg')
questions     = read_lines("Questions",'Question')

training_set = neg_sentiment + questions +  pos_sentiment

#Train the data
cl = NaiveBayesClassifier(training_set)

# Classify some text
print(cl.classify("iflix .com သွား ပြီး ဆွဲ မ ရ ဘူး ။".decode('utf8')) ) # "neg"
print(cl.classify("ကျန် တာ တွေ fb, you tube, Online video တွေ သုံး ရင် အားလုံး က အဆင်ပြေ ပါ တယ် download speed 2 MB လောက် ထိ တက် ပါ တယ် ".decode('utf8')))   # "pos"
print(cl.classify("လက်ဆောင်ပေး မဲ အ စားနယ် တိုင် ကို လိုင်းကောင်း အောင်လုပ် ပေး ပါလား မင်း တို့ လူကြီး တွေ ကို ပြော လိုက်ပါ ".decode('utf8'))) #quesion


# Compute accuracy
print("Accuracy: {0}".format(cl.accuracy(test)))

# Show 5 most informative features
cl.show_informative_features(10)
