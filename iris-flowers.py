from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

class ScrappyKNN():
    def fit(self, features_train,labels_train):
        self.features_train = features_train
        self.labels_train = labels_train


    def predict(self,features_test):
        prediction = []
        for item in features_test:
            #determine which other point is closest and use that to set label value
            predictions.append(label)

        return predictions

iris = datasets.load_iris()

features = iris.data
labels = iris.target

features_train,features_test,labels_train,labels_test = train_test_split(features,labels,test_size=.5)


my_classifier = KNeighborsClassifier()
my_classifier.fit(features_train, labels_train)

prediction = my_classifier.predict(features_test)

print(accuracy_score(labels_test, prediction))

#we want to create an instance of a versicolor flower to see if our model has lernt to identify this flowers from the dataset given

iris1 = [[4.7,2.5,3.1,1.2]]
iris_prediction = my_classifier.predict(iris1)

if iris_prediction[0]== 0 :
    print ("Setosa")
if iris_prediction[0]== 1:
    print ( "versicolor")
if iris_prediction[0]==2:
    print ("virginica")
